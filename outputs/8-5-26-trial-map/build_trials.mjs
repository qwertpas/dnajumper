import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const repo = "/Users/chris/Code/dnajumper";
const day = path.join(repo, "data/8-5-26");
const output = path.join(day, "trials.csv");
const preview = path.join(repo, "outputs/8-5-26-trial-map/trials-preview.png");

const trials = [
  "r7_v10_pole",
  "r7_v10_pole_2",
  "r7_v10_pole_3",
  "r7_v10_pole_4",
  "r7_v10_pole_5",
  "r7_v9_pole_1",
  "r7_v9_pole_2",
  "r7_v9_pole_3",
  "r7_v10_pole_6",
  "r7_v10_pole_7",
  "r7_v10_pole_8",
  "r7_v10_pole_9",
  "r7_v9_pole_4",
  "r7_v8_pole_1",
  "r7_v8_pole_2",
  "r7_v8_pole_3",
  "r7_v7_pole_0_noslowmo",
  "r7_v7_pole_2",
  "r7_v7_pole_3",
  "r7_v7_pole_4",
  "r7_v6_pole_1",
  "r7_v6_pole_2",
  "r7_v6_pole_3",
  "r7_v10_nopole_1",
];

const customNotes = {
  r7_v10_pole_7: "phone match inferred from slowmo timestamp",
  r7_v9_pole_4: "missing force CSV; wall clock time from motor file creation; duplicate extended slowmo exports moved to Trash",
  r7_v8_pole_1: "missing force CSV; wall clock time from motor file creation",
  r7_v7_pole_0_noslowmo: "phone match inferred by chronology (medium confidence)",
};

const exists = async (file) => {
  try {
    await fs.access(file);
    return true;
  } catch {
    return false;
  }
};

const localIso = (date) => {
  const pad = (value, width = 2) => String(value).padStart(width, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
    + `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}.${pad(date.getMilliseconds(), 3)}`;
};

const firstForceTime = async (file) => {
  const text = await fs.readFile(file, "utf8");
  const firstRow = text.split(/\r?\n/, 3)[1];
  if (!firstRow) throw new Error(`No data rows in ${file}`);
  return firstRow.split(",", 1)[0];
};

const rows = [];
for (const trial of trials) {
  const motor = path.join(day, "motor", `${trial}_motor.csv`);
  const force = path.join(day, "force", `${trial}_mindaq.csv`);
  const slowmo = path.join(day, "slowmo", `${trial}.mp4`);
  const phone = path.join(day, "phone", `${trial}.MOV`);
  if (!(await exists(motor))) throw new Error(`Missing motor CSV: ${motor}`);

  const notes = [];
  let wallClockTime;
  if (await exists(force)) {
    wallClockTime = await firstForceTime(force);
  } else {
    wallClockTime = localIso((await fs.stat(motor)).birthtime);
  }
  if (!(await exists(slowmo))) notes.push("missing slowmo vid");
  if (!(await exists(phone))) notes.push("missing phone vid");
  if (customNotes[trial]) notes.push(customNotes[trial]);
  rows.push([trial, wallClockTime, notes.join("; ")]);
}

rows.push([
  "UNASSIGNED_2026-08-05_14-18-54",
  "2026-08-05T14:18:54.000",
  "unassigned recording with no motor or force CSV; slowmo vid_2026-08-05_14-18-54.mp4; phone 2026-08-05_14-18-53.MOV; files left timestamp-named",
]);

const csvEscape = (value) => {
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
const csvRows = [["trial name", "wall clock time", "notes"], ...rows];
const csvText = `${csvRows.map((row) => row.map(csvEscape).join(",")).join("\n")}\n`;

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Trials");
sheet.showGridLines = false;
sheet.getRange(`A1:C${csvRows.length}`).values = csvRows;
sheet.freezePanes.freezeRows(1);
sheet.getRange("A1:C1").format = {
  fill: "#1F2937",
  font: { bold: true, color: "#FFFFFF" },
  rowHeight: 26,
};
sheet.getRange(`A2:C${csvRows.length}`).format = {
  borders: { preset: "insideHorizontal", style: "thin", color: "#D1D5DB" },
  verticalAlignment: "top",
};
sheet.getRange(`A1:C${csvRows.length}`).format.wrapText = true;
sheet.getRange(`A1:A${csvRows.length}`).format.columnWidth = 34;
sheet.getRange(`B1:B${csvRows.length}`).format.columnWidth = 27;
sheet.getRange(`C1:C${csvRows.length}`).format.columnWidth = 84;
sheet.getRange(`A${csvRows.length}:C${csvRows.length}`).format = {
  fill: "#FEF3C7",
  font: { color: "#92400E" },
};

const inspection = await workbook.inspect({
  kind: "table",
  range: `Trials!A1:C${csvRows.length}`,
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 3,
});
console.log(inspection.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const rendered = await workbook.render({ sheetName: "Trials", range: `A1:C${csvRows.length}`, scale: 1.2 });
await fs.writeFile(preview, new Uint8Array(await rendered.arrayBuffer()));
await fs.writeFile(output, csvText, "utf8");
console.log(JSON.stringify({ output, preview, rows: rows.length }));
