// Сборка черновика диссертации из content.txt в .docx (ГОСТ Р 7.0.11-2011)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, HeadingLevel, WidthType, ShadingType, BorderStyle, LevelFormat,
  Header, PageNumber, TableOfContents, TabStopType, HighlightColor,
} = require("docx");

const HERE = __dirname;  // content.txt и pipeline.png лежат рядом
const RESULTS = "/home/shironoir/myopia/results";
const OUT = process.argv[2];
const FONT = "Times New Roman";
const W = 9638;              // ширина текста: A4, поля 20 мм
const BODY = 28;             // 14 пт
const SMALL = 24;            // 12 пт — таблицы, подписи, комментарии

// ---------- инлайн-разметка ----------
function runs(text, base = {}) {
  const out = [];
  const re = /(\[\[TODO:[\s\S]*?\]\](?!\])|\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  const plain = (t) => t && out.push(new TextRun({ text: t, ...base }));
  while ((m = re.exec(text))) {
    plain(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith("[[TODO:")) {
      out.push(new TextRun({ text: "[TODO:" + t.slice(7, -2).replace(/`/g, "") + "]", ...base, bold: true, color: "C00000", highlight: HighlightColor.YELLOW }));
    } else if (t.startsWith("**")) {
      out.push(...runs(t.slice(2, -2), { ...base, bold: true }));
    } else {
      out.push(new TextRun({ text: t.slice(1, -1), ...base, font: "Consolas", size: Math.round((base.size || BODY) * 0.85) }));
    }
    last = m.index + t.length;
  }
  plain(text.slice(last));
  return out;
}

const para = (text) => new Paragraph({
  children: runs(text), alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 709 },
});
const bullet = (text) => new Paragraph({
  children: runs(text), numbering: { reference: "dash", level: 0 }, alignment: AlignmentType.JUSTIFIED,
});
const noteBox = (text) => new Paragraph({
  children: runs(text, { size: SMALL }), alignment: AlignmentType.JUSTIFIED,
  shading: { type: ShadingType.CLEAR, fill: "FFF4CE", color: "auto" },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: "E0A800", space: 6 } },
  spacing: { before: 120, after: 120, line: 276 }, indent: { left: 170, right: 113 },
});
const unnumberedHeading = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_1, alignment: AlignmentType.LEFT, children: [new TextRun(t)],
});
const chapterHeading = unnumberedHeading;
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(t) });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: runs(t) });
const tableTitle = (t) => new Paragraph({
  children: runs(t, { size: SMALL }), keepNext: true, spacing: { before: 200, after: 60, line: 276 },
});
const equation = (f, num) => new Paragraph({
  tabStops: [{ type: TabStopType.CENTER, position: Math.round(W / 2) }, { type: TabStopType.RIGHT, position: W }],
  spacing: { before: 120, after: 120 },
  children: [new TextRun({ text: "\t" + f, italics: true }), new TextRun({ text: "\t" + num })],
});

const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(rows) {
  const ncol = rows[0].length;
  const len = Array(ncol).fill(0);
  rows.forEach((r) => r.forEach((c, j) => (len[j] = Math.max(len[j], Math.min(c.length, 60)))));
  let widths = len.map((l) => Math.max(l, 6));
  const s = widths.reduce((a, b) => a + b, 0);
  widths = widths.map((x) => Math.floor((x / s) * W));
  widths[ncol - 1] += W - widths.reduce((a, b) => a + b, 0);
  const cell = (t, w, hdr) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: hdr ? { type: ShadingType.CLEAR, fill: "E7E6E6", color: "auto" } : undefined,
    margins: { top: 30, bottom: 30, left: 80, right: 80 },
    children: [new Paragraph({ spacing: { line: 240 }, alignment: hdr ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: runs(t, { size: SMALL, bold: hdr || undefined }) })],
  });
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0, cantSplit: true,
      children: r.map((c, j) => cell(c, widths[j], i === 0)) })),
  });
}
const spacer = () => new Paragraph({ spacing: { after: 0, line: 240 }, children: [] });

function figure(spec) {
  const [file, wpx, hpx, target, cap] = spec.split("|");
  const p = fs.existsSync(path.join(HERE, file)) ? path.join(HERE, file) : path.join(RESULTS, file);
  const w = Number(target), h = Math.round(w * Number(hpx) / Number(wpx));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(p), transformation: { width: w, height: h },
        altText: { title: cap, description: cap, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200, line: 276 }, children: runs(cap, { size: SMALL }) }),
  ];
}

function thresholdsTable() {
  const lines = fs.readFileSync(path.join(RESULTS, "threshold_analysis.csv"), "utf8").trim().split(/\r?\n/).slice(1);
  const c = (x) => x.replace(".", ",");
  return table([["Порог", "Чувствительность", "Специфичность", "Точность", "F1", "TP", "FP", "FN", "TN"],
    ...lines.map((l) => { const v = l.split(","); return [c(Number(v[0]).toFixed(2)), c(v[1]), c(v[2]), c(v[3]), c(v[4]), v[5], v[6], v[7], v[8]]; })]);
}

// ---------- разбор content.txt ----------
const src = fs.readFileSync(path.join(HERE, "content.txt"), "utf8").split(/\r?\n/);
const body = [];
let buf = [], tbl = null, refNo = 0;
const flush = () => { if (buf.length) { body.push(para(buf.join(" "))); buf = []; } };
const flushTable = () => { if (tbl) { body.push(table(tbl), spacer()); tbl = null; } };

for (const raw of src) {
  const line = raw.trimEnd();
  if (line.startsWith("|")) {
    flush();
    const cells = line.replace(/^\|/, "").replace(/\|$/, "").split("|").map((s) => s.trim());
    (tbl = tbl || []).push(cells);
    continue;
  }
  flushTable();
  if (!line.trim()) { flush(); continue; }
  flush();
  if (line === "TOC") {
    body.push(new Paragraph({ alignment: AlignmentType.CENTER, pageBreakBefore: true, spacing: { after: 240 },
      children: [new TextRun({ text: "ОГЛАВЛЕНИЕ", bold: true })] }));
    body.push(new TableOfContents("Оглавление", { hyperlink: true, headingStyleRange: "1-2" }));
  } else if (line.startsWith("#U ")) body.push(unnumberedHeading(line.slice(3)));
  else if (line.startsWith("#C ")) body.push(chapterHeading(line.slice(3)));
  else if (line.startsWith("#T ")) body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 240 },
    children: runs(line.slice(3), { bold: true }) }));
  else if (line.startsWith("#M ")) body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: runs(line.slice(3)) }));
  else if (line.startsWith("### ")) body.push(h3(line.slice(4)));
  else if (line.startsWith("## ")) body.push(h2(line.slice(3)));
  else if (line.startsWith("> ")) body.push(noteBox(line.slice(2)));
  else if (line.startsWith("- ")) body.push(bullet(line.slice(2)));
  else if (line.startsWith("T: ")) body.push(tableTitle(line.slice(3)));
  else if (line.startsWith("F: ")) body.push(...figure(line.slice(3)));
  else if (line.startsWith("E: ")) { const e = line.slice(3), k = e.lastIndexOf("|"); body.push(equation(e.slice(0, k), e.slice(k + 1))); }
  else if (line.startsWith("R: ")) {
    refNo += 1;
    body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 567, hanging: 567 },
      children: [new TextRun(`${refNo}. `), ...runs(line.slice(3))] }));
  } else if (line.startsWith("C: ") || line === "C:") {
    body.push(new Paragraph({ spacing: { line: 240, after: 0 }, keepLines: true,
      children: [new TextRun({ text: line.slice(3).replace(/ /g, "\u00A0"), font: "Consolas", size: 18 })] }));
  } else if (line === "!THRESHOLDS") body.push(thresholdsTable(), spacer());
  else buf.push(line);
}
flush(); flushTable();

const doc = new Document({
  creator: "Халилов В. Р.",
  title: "Статья: менделевская рандомизация, образование и миопия",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: BODY }, paragraph: { spacing: { line: 360, after: 0 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: BODY, bold: true },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: BODY, bold: true },
        paragraph: { spacing: { before: 360, after: 180 }, indent: { firstLine: 709 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: BODY, bold: true, italics: true },
        paragraph: { spacing: { before: 240, after: 120 }, indent: { firstLine: 709 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "dash", levels: [{ level: 0, format: LevelFormat.BULLET, text: "–",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1069, hanging: 360 } } } }] }] },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134, header: 567 } },
    },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT] })] })] }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("written", OUT, "refs:", refNo); });
