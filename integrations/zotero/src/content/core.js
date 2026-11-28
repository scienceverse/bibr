/* The parts of the plugin that do not touch Zotero: the bibr HTTP client and
 * the mapping from a bibr export to CSL-JSON. Loaded into the plugin scope by
 * bootstrap.js and required by the Node tests. */

var BibrCore = (function () {
	const MIME_TYPES = {
		pdf: "application/pdf",
		docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
		epub: "application/epub+zip",
		html: "text/html",
		htm: "text/html",
		xml: "application/xml",
		nxml: "application/xml",
	};

	class BibrError extends Error {
		constructor(message, status) {
			super(message);
			this.name = "BibrError";
			this.status = status;
		}
	}

	function text(value) {
		if (value === null || value === undefined) return null;
		const s = String(value).trim();
		return s ? s : null;
	}

	function bareDOI(value) {
		const s = text(value);
		if (!s) return null;
		return s.replace(/^(?:https?:\/\/(?:dx\.)?doi\.org\/|doi:\s*)/i, "");
	}

	function dateParts(value) {
		const s = text(value);
		if (!s) return null;
		const m = /^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?/.exec(s);
		if (!m) return { raw: s };
		const parts = [Number(m[1])];
		if (m[2]) parts.push(Number(m[2]));
		if (m[2] && m[3]) parts.push(Number(m[3]));
		return { "date-parts": [parts] };
	}

	function cslName(author) {
		const family = text(author.family);
		const given = text(author.given);
		if (family || given) {
			const name = {};
			if (family) name.family = family;
			if (given) name.given = given;
			if (text(author.suffix)) name.suffix = text(author.suffix);
			return name;
		}
		const literal = text(author.literal);
		return literal ? { literal } : null;
	}

	/**
	 * The paper itself as one CSL-JSON item, from a bibr export (schema 12).
	 * Only what bibr read from the paper (`metadata`, `author`); registry
	 * matches (`metadata_match`) are never used.
	 */
	function paperToCSL(exp) {
		const m = (exp && exp.metadata) || {};
		const item = {
			id: text(exp && exp.paper_id) || "paper",
			type: text(m.journal) ? "article-journal" : "article",
		};
		const put = (key, value) => {
			const v = text(value);
			if (v) item[key] = v;
		};
		put("title", m.title);
		put("abstract", m.abstract);
		put("container-title", m.journal);
		put("volume", m.volume);
		put("issue", m.issue);
		const first = text(m.first_page);
		const last = text(m.last_page);
		if (first) item.page = last && last !== first ? `${first}-${last}` : first;
		put("ISSN", m.issn);
		put("publisher", m.publisher);
		put("language", m.language);
		put("DOI", bareDOI(m.doi));
		put("PMID", m.pmid);
		put("PMCID", m.pmcid);
		const issued = dateParts(m.published_date || m.published);
		if (issued) item.issued = issued;
		const authors = ((exp && exp.author) || []).map(cslName).filter(Boolean);
		if (authors.length) item.author = authors;
		if (text(m.arxiv)) item.note = `arXiv: ${text(m.arxiv)}`;
		return item;
	}

	/**
	 * Which of `incoming`'s fields to write onto an item whose values are
	 * `current`: empty ones only, or every differing one with `overwrite`.
	 */
	function planUpdates(current, incoming, overwrite) {
		const updates = [];
		for (const [field, value] of Object.entries(incoming)) {
			const to = text(value);
			if (!to) continue;
			const from = text(current[field]);
			if (from === to) continue;
			if (from && !overwrite) continue;
			updates.push({ field, from, to });
		}
		return updates;
	}

	function referencesCollectionName(title, filename) {
		let name = text(title) || text(filename) || "Untitled";
		if (name.length > 80) name = name.slice(0, 79).trimEnd() + "…";
		return `${name} – references (bibr)`;
	}

	function mimeType(filename, fallback) {
		const ext = (/\.([A-Za-z0-9]+)$/.exec(filename || "") || [])[1];
		return (ext && MIME_TYPES[ext.toLowerCase()]) || fallback || "application/octet-stream";
	}

	async function errorDetail(response) {
		try {
			const body = await response.json();
			if (body && body.detail) {
				return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
			}
		} catch (e) {
			// Not JSON: fall through to the status line.
		}
		return `HTTP ${response.status}`;
	}

	/**
	 * A client for bibr serve's async jobs API: POST /papers/jobs, poll
	 * /papers/jobs/{id}, then fetch /result and /references?format=csl-json.
	 * `fetch`, `FormData` and `Blob` are injected so the same code runs in
	 * Zotero and in Node.
	 */
	class BibrClient {
		constructor({ baseURL, apiKey, fetch, FormData, Blob, sleep, pollSeconds = 2 }) {
			this.baseURL = String(baseURL || "").replace(/\/+$/, "");
			this.apiKey = text(apiKey);
			this.fetch = fetch;
			this.FormData = FormData;
			this.Blob = Blob;
			this.sleep = sleep || ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
			this.pollSeconds = pollSeconds;
		}

		headers(extra) {
			const headers = Object.assign({}, extra);
			if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
			return headers;
		}

		async request(path, options = {}) {
			let response;
			try {
				response = await this.fetch(this.baseURL + path, {
					method: options.method || "GET",
					body: options.body,
					headers: this.headers(options.headers),
				});
			} catch (e) {
				throw new BibrError(`Cannot reach the bibr server at ${this.baseURL} (${e.message})`);
			}
			if (response.status === 401) {
				throw new BibrError("The bibr server wants an API key: set it in the bibr preferences", 401);
			}
			if (response.status === 404 && path === "/papers/jobs") {
				throw new BibrError(
					"This bibr server has no jobs API (JOBS_ENABLED=false, or not a bibr server)",
					404
				);
			}
			return response;
		}

		async submit(bytes, filename, contentType) {
			const form = new this.FormData();
			const blob = new this.Blob([bytes], { type: contentType || mimeType(filename) });
			form.append("file", blob, filename);
			const response = await this.request("/papers/jobs", { method: "POST", body: form });
			if (response.status !== 202) {
				throw new BibrError(`bibr refused the file: ${await errorDetail(response)}`, response.status);
			}
			const body = await response.json();
			return body.job_id;
		}

		async wait(jobId, { timeoutSeconds = 900, onStatus } = {}) {
			const deadline = Date.now() + timeoutSeconds * 1000;
			for (;;) {
				const response = await this.request(`/papers/jobs/${encodeURIComponent(jobId)}`);
				if (!response.ok) {
					throw new BibrError(`Lost the bibr job: ${await errorDetail(response)}`, response.status);
				}
				const status = await response.json();
				if (onStatus) onStatus(status.status);
				if (status.status === "succeeded") return status;
				if (status.status === "failed") {
					const detail = (status.error && status.error.detail) || "unknown error";
					throw new BibrError(`bibr could not read the file: ${detail}`);
				}
				if (Date.now() > deadline) {
					throw new BibrError(`bibr took longer than ${timeoutSeconds} s`);
				}
				await this.sleep(this.pollSeconds * 1000);
			}
		}

		async json(path) {
			const response = await this.request(path);
			if (!response.ok) {
				throw new BibrError(`bibr: ${await errorDetail(response)}`, response.status);
			}
			return response.json();
		}

		result(jobId) {
			return this.json(`/papers/jobs/${encodeURIComponent(jobId)}/result`);
		}

		/** The reference list as CSL-JSON, values as printed in the paper. */
		references(jobId) {
			return this.json(
				`/papers/jobs/${encodeURIComponent(jobId)}/references?format=csl-json&prefer=printed`
			);
		}

		/** Upload one file and return `{ export, references }`. */
		async extract(bytes, filename, contentType, { timeoutSeconds, onStatus, references = true } = {}) {
			const jobId = await this.submit(bytes, filename, contentType);
			await this.wait(jobId, { timeoutSeconds, onStatus });
			const exp = await this.result(jobId);
			const refs = references ? await this.references(jobId) : [];
			return { export: exp, references: refs };
		}
	}

	return {
		BibrClient,
		BibrError,
		bareDOI,
		dateParts,
		mimeType,
		paperToCSL,
		planUpdates,
		referencesCollectionName,
	};
})();

if (typeof module !== "undefined") module.exports = BibrCore;
