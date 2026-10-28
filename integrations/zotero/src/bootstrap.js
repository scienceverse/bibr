var BibrZotero;

async function startup({ id, version, rootURI }) {
	Services.scriptloader.loadSubScript(rootURI + "content/core.js");
	Services.scriptloader.loadSubScript(rootURI + "content/bibr.js");
	BibrZotero.init({ id, version, rootURI });
	await Zotero.PreferencePanes.register({
		pluginID: id,
		label: "bibr",
		src: rootURI + "preferences.xhtml",
	});
	BibrZotero.addToAllWindows();
}

function onMainWindowLoad({ window }) {
	BibrZotero.addToWindow(window);
}

function onMainWindowUnload({ window }) {
	BibrZotero.removeFromWindow(window);
}

function shutdown() {
	BibrZotero.removeFromAllWindows();
	BibrZotero = undefined;
	BibrCore = undefined;
}

function install() {}

function uninstall() {}
