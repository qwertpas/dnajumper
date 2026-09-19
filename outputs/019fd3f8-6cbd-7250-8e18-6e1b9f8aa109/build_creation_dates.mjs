import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const sourceDir = "/Users/chris/Code/dnajumper/data/8-5-26";
const outputPath = path.join(sourceDir, "creation_dates.csv");
const previewPath = "/Users/chris/Code/dnajumper/outputs/019fd3f8-6cbd-7250-8e18-6e1b9f8aa109/creation_dates_preview.png";

const names = (await fs.readdir(sourceDir))
  .filter((name) => name.endsWith(".csv") && name !== path.basename(outputPath));

const trials = new Map();
for (const name of names) {
  const fullPath = path.join(sourceDir, name);
  const stat = await fs.stat(fullPath);
  const trial = name.replace(/_(mindaq|motor)\.csv$/, "").replace(/\.csv$/, "");
  const created = stat.birthtime;
  const current = trials.get(trial);
  if (!current || created < current) trials.set(trial, created);
}

const rows = [...trials.entries()]
  .sort((a, b) => a[1] - b[1] || a[0].localeCompare(b[0]))
  .map(([trial, created]) => [
    trial,
    new Intl.DateTimeFormat("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
      timeZone: "America/Los_Angeles",
    }).format(created),
  ]);

const escapeCsv = (value) => {
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
const csv = [["trial", "time"], ...rows]
  .map((row) => row.map(escapeCsv).join(","))
  .join("\n") + "\n";

const workbook = await Workbook.fromCSV(csv, { sheetName: "Creation dates" });
const sheet = workbook.worksheets.getItem("Creation dates");
sheet.getRange("A1:B1").format = {
  fill: "#E7E6E6",
  font: { bold: true, color: "#000000" },
  borders: { preset: "all", style: "thin", color: "#D9D9D9" },
};
sheet.getRange(`A2:B${rows.length + 1}`).format.borders = {
  preset: "all",
  style: "thin",
  color: "#D9D9D9",
};
sheet.getRange(`A1:A${rows.length + 1}`).format.columnWidth = 28;
sheet.getRange(`B1:B${rows.length + 1}`).format.columnWidth = 12;
sheet.getRange(`B2:B${rows.length + 1}`).format.horizontalAlignment = "right";

const check = await workbook.inspect({
  kind: "table",
  range: `Creation dates!A1:B${rows.length + 1}`,
  include: "values,formulas",
  tableMaxRows: rows.length + 1,
  tableMaxCols: 2,
});
console.log(check.ndjson);

const preview = await workbook.render({
  sheetName: "Creation dates",
  range: `A1:B${rows.length + 1}`,
  scale: 2,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
await fs.writeFile(outputPath, csv, "utf8");
console.log(JSON.stringify({ outputPath, previewPath, sourceFiles: names.length, trials: rows.length }));
