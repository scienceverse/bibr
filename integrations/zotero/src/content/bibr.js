/* The Zotero side of the plugin: the item context-menu entry, reading the
 * attachment, and writing bibr's answer into the library. */

BibrZotero = {
	id: null,
	version: null,
	rootURI: null,
	addedElementIDs: ["bibr-itemmenu-separator", "bibr-itemmenu-extract"],

	init({ id, version, rootURI }) {
		this.id = id;
		this.version = version;
		this.rootURI = rootURI;
	},

	log(msg) {
		Zotero.debug("bibr: " + msg);
	},

	pref(name) {
		return Zotero.Prefs.get("extensions.bibr." + name, true);
	},

	addToWindow(window) {
		const doc = window.document;
		const menu = doc.getElementById("zotero-itemmenu");
		if (!menu) return;
		const separator = doc.createXULElement("menuseparator");
		separator.id = "bibr-itemmenu-separator";
		const item = doc.createXULElement("menuitem");
		item.id = "bibr-itemmenu-extract";
		item.setAttribute("label", "Fill metadata and import references with bibr");
		item.addEventListener("command", () => {
			this.runOnSelection(window).catch((e) => Zotero.logError(e));
		});
		menu.appendChild(separator);
		menu.appendChild(item);
	},

	addToAllWindows() {
		for (const win of Zotero.getMainWindows()) {
			if (win.ZoteroPane) this.addToWindow(win);
		}
	},

	removeFromWindow(window) {
		for (const id of this.addedElementIDs) {
			window.document.getElementById(id)?.remove();
		}
	},

	removeFromAllWindows() {
		for (const win of Zotero.getMainWindows()) {
			if (win.ZoteroPane) this.removeFromWindow(win);
		}
	},

	client() {
		return new BibrCore.BibrClient({
			baseURL: this.pref("serverURL"),
			apiKey: this.pref("apiKey"),
			fetch: (...args) => fetch(...args),
			FormData,
			Blob,
			sleep: (ms) => Zotero.Promise.delay(ms),
		});
	},

	async runOnSelection(window) {
		const pane = window.ZoteroPane;
		const items = pane.getSelectedItems().filter((i) => i.isRegularItem() || i.isAttachment());
		const progress = new Zotero.ProgressWindow({ closeOnClick: false });
		progress.changeHeadline("bibr");
		if (!items.length) {
			progress.addDescription("Select an item with a PDF, or the PDF itself.");
			progress.show();
			progress.startCloseTimer(5000);
			return;
		}
		progress.show();
		const collection = pane.getSelectedCollection();
		for (const item of items) {
			const line = new progress.ItemProgress(item.itemType, item.getDisplayTitle());
			try {
				const summary = await this.processItem(item, { window, collection, line });
				line.setText(`${item.getDisplayTitle()}: ${summary}`);
				line.setProgress(100);
			} catch (e) {
				this.log(e.stack || e);
				line.setError();
				line.setText(`${item.getDisplayTitle()}: ${e.message}`);
			}
		}
		progress.startCloseTimer(8000);
	},

	async processItem(item, { window, collection, line }) {
		if (!Zotero.Libraries.get(item.libraryID).editable) {
			throw new Error("this library is read-only");
		}
		let source = item.isRegularItem() ? item : item.parentItem || null;
		const attachment = item.isAttachment() ? item : await item.getBestAttachment();
		const path = attachment && (await attachment.getFilePathAsync());
		if (!path) throw new Error("no stored file to send (a PDF, Word, EPUB or HTML attachment)");

		const filename = attachment.attachmentFilename || PathUtils.filename(path);
		const bytes = await IOUtils.read(path);
		line.setText(`${filename}: sending to bibr…`);
		const importReferences = this.pref("importReferences");
		const answer = await this.client().extract(bytes, filename, attachment.attachmentContentType, {
			timeoutSeconds: this.pref("timeoutSeconds"),
			references: importReferences,
			onStatus: (status) => line.setText(`${filename}: bibr is reading it (${status})…`),
		});

		const csl = BibrCore.paperToCSL(answer.export);
		const done = [];
		if (source) {
			const n = await this.fillItem(source, csl);
			done.push(n ? `filled ${n} field${n === 1 ? "" : "s"}` : "metadata already complete");
		} else {
			source = await this.createParent(attachment, csl);
			done.push("created the parent item");
		}

		if (importReferences) {
			const n = await this.importReferences(source, answer.references, { window, collection, csl });
			if (n !== null) done.push(`imported ${n} reference${n === 1 ? "" : "s"}`);
		}
		return done.join(", ");
	},

	/** A new, unsaved Zotero item holding bibr's reading of the paper. */
	itemFromCSL(csl) {
		const item = new Zotero.Item("journalArticle");
		const fromCSL = (Zotero.Utilities.Item || Zotero.Utilities).itemFromCSLJSON;
		fromCSL(item, csl);
		return item;
	},

	/** Field name on `target`'s item type for `fieldID` of `source`'s type. */
	targetField(source, target, fieldID) {
		if (Zotero.ItemFields.isValidForType(fieldID, target.itemTypeID)) {
			return Zotero.ItemFields.getName(fieldID);
		}
		const baseID = Zotero.ItemFields.getBaseIDFromTypeAndField(source.itemTypeID, fieldID) || fieldID;
		const mapped = Zotero.ItemFields.getFieldIDFromTypeAndBase(target.itemTypeID, baseID);
		return mapped ? Zotero.ItemFields.getName(mapped) : null;
	},

	async fillItem(target, csl) {
		const incoming = this.itemFromCSL(csl);
		const overwrite = this.pref("overwrite");
		const current = {};
		const values = {};
		const extra = {};
		for (const fieldID of Zotero.ItemFields.getItemTypeFields(incoming.itemTypeID)) {
			const name = this.targetField(incoming, target, fieldID);
			const value = incoming.getField(Zotero.ItemFields.getName(fieldID));
			if (!name || !value) continue;
			// Extra holds the user's own notes: fill it when empty, never replace it.
			const into = name === "extra" ? extra : values;
			into[name] = value;
			current[name] = target.getField(name);
		}
		const updates = BibrCore.planUpdates(current, values, overwrite).concat(
			BibrCore.planUpdates(current, extra, false)
		);
		for (const { field, to } of updates) target.setField(field, to);
		let changed = updates.length;
		const creators = incoming.getCreators();
		if (creators.length && (target.numCreators() === 0 || overwrite)) {
			target.setCreators(creators);
			changed += 1;
		}
		if (changed) await target.saveTx();
		return changed;
	},

	async createParent(attachment, csl) {
		const parent = this.itemFromCSL(csl);
		parent.libraryID = attachment.libraryID;
		parent.setCollections(attachment.getCollections());
		await Zotero.DB.executeTransaction(async () => {
			await parent.save();
			attachment.parentID = parent.id;
			await attachment.save();
		});
		return parent;
	},

	async findByDOI(libraryID, doi) {
		const sql =
			"SELECT itemID FROM items JOIN itemData USING (itemID) JOIN itemDataValues USING (valueID) " +
			"WHERE libraryID=? AND fieldID=? AND LOWER(value)=? " +
			"AND itemID NOT IN (SELECT itemID FROM deletedItems) LIMIT 1";
		const id = await Zotero.DB.valueQueryAsync(sql, [
			libraryID,
			Zotero.ItemFields.getID("DOI"),
			doi.toLowerCase(),
		]);
		return id ? Zotero.Items.getAsync(id) : null;
	},

	/**
	 * Put the reference list in a new collection (under the one selected, if
	 * any), reusing library items with the same DOI, and relate each to the
	 * paper. Values are as printed in the paper; bibr's reference check, when
	 * it ran, is in each item's Extra. Returns null when the user cancels.
	 */
	async importReferences(source, references, { window, collection, csl }) {
		if (!references.length) return 0;
		const libraryID = source.libraryID;
		const parentID = collection && collection.libraryID === libraryID ? collection.id : null;
		const name = BibrCore.referencesCollectionName(csl.title || source.getDisplayTitle());
		const siblings = parentID
			? Zotero.Collections.getByParent(parentID)
			: Zotero.Collections.getByLibrary(libraryID);
		if (siblings.some((c) => c.name === name)) {
			const again = Services.prompt.confirm(
				window,
				"bibr",
				`"${name}" already exists. Import the references again into a second collection?`
			);
			if (!again) return null;
		}

		const byDOI = new Map();
		for (const ref of references) {
			const doi = BibrCore.bareDOI(ref.DOI);
			if (doi && !byDOI.has(doi.toLowerCase())) {
				byDOI.set(doi.toLowerCase(), await this.findByDOI(libraryID, doi));
			}
		}
		await Zotero.DB.executeTransaction(async () => {
			const target = new Zotero.Collection({ libraryID, name });
			if (parentID) target.parentID = parentID;
			await target.save();
			for (const ref of references) {
				const doi = (BibrCore.bareDOI(ref.DOI) || "").toLowerCase();
				let item = doi ? byDOI.get(doi) : null;
				if (!item) {
					item = this.itemFromCSL(ref);
					item.libraryID = libraryID;
				}
				// Two references with one DOI, or a paper citing itself, map to one item.
				if (doi) byDOI.set(doi, item);
				item.addToCollection(target.id);
				if (item.id !== source.id) item.addRelatedItem(source);
				await item.save();
				if (item.id !== source.id) source.addRelatedItem(item);
			}
			await source.save();
		});
		return references.length;
	},
};
