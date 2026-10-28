// node --test integrations/zotero/test/
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");

const core = require("../src/content/core.js");

const FIXTURE = path.join(
	__dirname,
	"..",
	"..",
	"..",
	"tests",
	"fixtures",
	"references",
	"enriched_check_12_3.json"
);
const exp = JSON.parse(fs.readFileSync(FIXTURE, "utf8"));

test("paperToCSL maps what bibr read from the paper", () => {
	const csl = core.paperToCSL(exp);
	assert.deepEqual(csl, {
		id: "demo",
		type: "article-journal",
		title: "A Demonstration Paper",
		abstract: "We demonstrate the export schema.",
		"container-title": "Journal of Demonstrations",
		volume: "12",
		issue: "3",
		page: "100-115",
		ISSN: "1234-5678",
		publisher: "Example Press",
		DOI: "10.1234/demo",
		issued: { "date-parts": [[2026, 1, 15]] },
		author: [{ family: "Smith", given: "Jane" }],
	});
});

test("paperToCSL never takes values from registry matches", () => {
	const bare = Object.assign({}, exp, {
		metadata: { title: "Only a title" },
		author: [],
	});
	const csl = core.paperToCSL(bare);
	assert.deepEqual(csl, { id: "demo", type: "article", title: "Only a title" });
});

test("paperToCSL handles literal names, partial dates, DOI URLs and arXiv ids", () => {
	const csl = core.paperToCSL({
		metadata: { doi: "https://doi.org/10.5/X", published: "2024-03", arxiv: "2401.00001", first_page: "7" },
		author: [{ literal: "Open Science Collaboration" }, { given: null, family: null }],
	});
	assert.equal(csl.DOI, "10.5/X");
	assert.deepEqual(csl.issued, { "date-parts": [[2024, 3]] });
	assert.equal(csl.page, "7");
	assert.equal(csl.note, "arXiv: 2401.00001");
	assert.deepEqual(csl.author, [{ literal: "Open Science Collaboration" }]);
	assert.deepEqual(core.dateParts("in press"), { raw: "in press" });
});

test("planUpdates fills empty fields only unless told to overwrite", () => {
	const current = { title: "Kept", DOI: "", volume: "12" };
	const incoming = { title: "New", DOI: "10.1/x", volume: "12", issue: "" };
	assert.deepEqual(core.planUpdates(current, incoming, false), [
		{ field: "DOI", from: null, to: "10.1/x" },
	]);
	assert.deepEqual(core.planUpdates(current, incoming, true), [
		{ field: "title", from: "Kept", to: "New" },
		{ field: "DOI", from: null, to: "10.1/x" },
	]);
});

test("referencesCollectionName stays readable", () => {
	assert.equal(core.referencesCollectionName("A Paper"), "A Paper – references (bibr)");
	const long = core.referencesCollectionName("x".repeat(200));
	assert.ok(long.length < 110);
	assert.equal(core.referencesCollectionName(null, "f.pdf"), "f.pdf – references (bibr)");
});

test("mimeType follows the file extension", () => {
	assert.equal(core.mimeType("a.PDF"), "application/pdf");
	assert.equal(core.mimeType("a.unknown", "text/plain"), "text/plain");
});

function fakeServer(routes) {
	const calls = [];
	const fetch = async (url, options) => {
		calls.push({ url, options });
		const { pathname, search } = new URL(url);
		const route = routes[`${options.method} ${pathname}${search}`];
		if (!route) return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
		const [status, body] = typeof route === "function" ? route() : route;
		return new Response(JSON.stringify(body), { status });
	};
	return { fetch, calls };
}

test("BibrClient submits, polls and fetches the result and references", async () => {
	let polls = 0;
	const server = fakeServer({
		"POST /papers/jobs": [202, { job_id: "j1", status: "queued" }],
		"GET /papers/jobs/j1": () => [200, { status: ++polls < 3 ? "running" : "succeeded" }],
		"GET /papers/jobs/j1/result": [200, exp],
		"GET /papers/jobs/j1/references?format=csl-json&prefer=printed": [200, [{ id: "r1", type: "book" }]],
	});
	const client = new core.BibrClient({
		baseURL: "http://bibr.test/",
		apiKey: "k",
		fetch: server.fetch,
		FormData,
		Blob,
		sleep: async () => {},
	});
	const seen = [];
	const answer = await client.extract(new Uint8Array([37, 80]), "a.pdf", "application/pdf", {
		onStatus: (s) => seen.push(s),
	});
	assert.equal(answer.export.paper_id, "demo");
	assert.deepEqual(answer.references, [{ id: "r1", type: "book" }]);
	assert.deepEqual(seen, ["running", "running", "succeeded"]);
	const upload = server.calls[0];
	assert.equal(upload.url, "http://bibr.test/papers/jobs");
	assert.equal(upload.options.headers.Authorization, "Bearer k");
	assert.equal(upload.options.body.get("file").name, "a.pdf");
});

test("BibrClient reports a failed job and a missing jobs API", async () => {
	const failing = fakeServer({
		"POST /papers/jobs": [202, { job_id: "j2" }],
		"GET /papers/jobs/j2": [200, { status: "failed", error: { detail: "not a paper" } }],
	});
	const client = new core.BibrClient({ baseURL: "http://b", fetch: failing.fetch, FormData, Blob });
	await assert.rejects(client.extract(new Uint8Array(1), "a.pdf"), /not a paper/);

	const old = new core.BibrClient({ baseURL: "http://b", fetch: fakeServer({}).fetch, FormData, Blob });
	await assert.rejects(old.extract(new Uint8Array(1), "a.pdf"), /no jobs API/);
});

const CONTRACT_URL = process.env.BIBR_CONTRACT_URL;

test("contract: the real bibr job routes", { skip: !CONTRACT_URL && "set BIBR_CONTRACT_URL" }, async () => {
	const client = new core.BibrClient({ baseURL: CONTRACT_URL, fetch, FormData, Blob, pollSeconds: 0.05 });
	const answer = await client.extract(Buffer.from("%PDF-1.4\n"), "demo.pdf", "application/pdf", {
		timeoutSeconds: 30,
	});
	assert.equal(core.paperToCSL(answer.export).title, "A Demonstration Paper");
	assert.equal(answer.references.length, exp.bib.length);
	const first = answer.references[0];
	assert.equal(first.title, "A Prior Study");
	// Printed-only: the record's volume (8) never replaces the printed one (7).
	assert.equal(first.volume, "7");
	assert.ok(first.author.length >= 2, "the printed author string is parsed into names");
});
