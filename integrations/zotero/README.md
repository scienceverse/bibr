# bibr for Zotero

A Zotero 7/8 plugin that sends an item's PDF (or Word, EPUB, HTML) to a bibr
server and writes the answer back into the library:

- **Metadata.** Fields the item leaves empty are filled from what bibr read on
  the paper: title, authors, abstract, journal, volume, issue, pages, date,
  DOI, ISSN, publisher. Fields that have a value are kept unless you tick
  *Replace fields that already have a value*. A PDF without a parent item gets
  one, as Zotero's *Create Parent Item* would.
- **References.** The reference list goes into a new collection
  (`<title> – references (bibr)`, under the collection you are in), and each
  reference is marked as related to the paper. A reference whose DOI is
  already in the library reuses that item.

Every value is as printed in the paper. The plugin asks for
`prefer=printed`, so bibr never fills in a missing DOI or completes a
reference from a Crossref record. When bibr's reference check ran (server
setting `CROSSREF_ENRICH`), what the record says differently is in the
reference's *Extra* field.

## Use

1. Run a bibr server. Locally: `bibr serve` (it listens on
   `http://127.0.0.1:8000`, the plugin's default). A hosted server works too;
   it needs the async jobs API (`JOBS_ENABLED`, on by default).
2. Install the plugin: *Tools → Plugins → gear menu → Install Plugin From
   File…*, then pick `bibr-zotero-<version>.xpi`.
3. Set the server URL, and the API key if the server has `AUTH_API_KEY`, in
   *Settings → bibr*.
4. Select one or more items, right-click, and choose *Fill metadata and import
   references with bibr*. A progress window shows each item's outcome.

## Build and test

```bash
python integrations/zotero/build.py          # → integrations/zotero/dist/bibr-zotero-0.1.0.xpi
node --test integrations/zotero/test/core.test.js
```

The contract test runs the plugin's HTTP client against bibr's real job
routes, with the pipeline replaced by the 12.3 reference fixture:

```bash
uv run python integrations/zotero/test/contract_server.py --port 8765 &
BIBR_CONTRACT_URL=http://127.0.0.1:8765 node --test integrations/zotero/test/core.test.js
```

`src/content/core.js` holds everything that does not touch Zotero (the
client and the export → CSL-JSON mapping) and is what the Node tests cover.
`src/content/bibr.js` is the Zotero side: the menu entry, reading the file,
and writing items and collections.
