// Extract the AI-Toolkit UI training baseline from a pinned image.
//
// Run inside the pinned AI-Toolkit image with Node. The UI builds a job by
// starting from ui/src/app/jobs/new/jobConfig.ts (model arch "flex1") and,
// when an architecture is selected, applying ui/src/app/jobs/new/utils.ts
// handleModelArchChange: low_vram falls back to false for an arch without the
// model.low_vram section, the previous arch's [unselected] defaults are
// restored, and the new arch's [selected] defaults are applied. Extension
// ui.tsx modules are transpiled and evaluated the way
// ui/src/extensions/modelArchs.ts does. This script repeats those steps and
// prints JSON; it never guesses a value it cannot evaluate.
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const ROOT = '/app/ai-toolkit';
const ts = require(path.join(ROOT, 'ui/node_modules/typescript'));

const TRAIN_KEYS = [
  'noise_scheduler', 'timestep_type', 'dtype', 'optimizer', 'optimizer_params', 'lr',
  'content_or_style', 'loss_type', 'unload_text_encoder', 'cache_text_embeddings',
];
const MODEL_KEYS = ['quantize', 'qtype', 'quantize_te', 'qtype_te', 'low_vram', 'model_kwargs'];
const DATASET_KEYS = ['cache_latents_to_disk'];

const sources = {};

function load(relative, shims) {
  const file = path.join(ROOT, relative);
  const text = fs.readFileSync(file, 'utf8');
  sources[relative] = crypto.createHash('sha256').update(text).digest('hex');
  const output = ts.transpileModule(text, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
      jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true,
    },
    fileName: file,
  }).outputText;
  const module = { exports: {} };
  const requireShim = (name) => {
    if (Object.prototype.hasOwnProperty.call(shims, name)) return shims[name];
    throw new Error(`${relative} imports unshimmed module ${name}`);
  };
  new Function('require', 'module', 'exports', output)(requireShim, module, module.exports);
  return module.exports;
}

const inert = new Proxy(function inert() { return null; }, { get: () => inert, apply: () => null });
const react = { __esModule: true, default: inert, createElement: () => null, Fragment: 'fragment' };
const jsxRuntime = { jsx: () => null, jsxs: () => null, Fragment: 'fragment' };

const defaultSamples = load('ui/src/helpers/defaultSamples.ts', { '@/types': {} });
const jobConfigModule = load('ui/src/app/jobs/new/jobConfig.ts', {
  '@/helpers/basic': { isMac: () => false },
  '@/helpers/defaultSamples': defaultSamples,
  '@/types': {},
});
const options = load('ui/src/app/jobs/new/options.tsx', {
  react, 'react/jsx-runtime': jsxRuntime, '@/types': {}, './jobConfig': jobConfigModule,
});
load('ui/src/app/jobs/new/utils.ts', { '@/types': {}, './options': options, '@/utils/basic': { objectCopy: (value) => JSON.parse(JSON.stringify(value)) } });

const shims = {
  react,
  'react/jsx-runtime': jsxRuntime,
  'react/jsx-dev-runtime': jsxRuntime,
  'next/link': { __esModule: true, default: inert },
  '@/helpers/defaultSamples': defaultSamples,
  '@/components/formInputs': inert,
  '@/types': {},
  '@/app/jobs/new/options': options,
};

// Later modules override earlier ones by arch name (modelArchs.ts).
const archs = new Map();
for (const dir of ['extensions_built_in', 'extensions']) {
  const base = path.join(ROOT, dir);
  if (!fs.existsSync(base)) continue;
  for (const pkg of fs.readdirSync(base).sort()) {
    for (const name of ['ui.tsx', 'ui.ts', 'ui.jsx', 'ui.js']) {
      const relative = path.join(dir, pkg, name);
      if (!fs.existsSync(path.join(ROOT, relative))) continue;
      const list = load(relative, shims).AI_TOOLKIT_UI_MODELS;
      if (!Array.isArray(list)) throw new Error(`${relative} does not export AI_TOOLKIT_UI_MODELS`);
      for (const entry of list) archs.set(entry.name, entry);
      break;
    }
  }
}

// Apply one UI default key ("config.process[0].a.b") to a process object.
function setPath(process, key, value) {
  const match = /^config\.process\[0\]\.(.+)$/.exec(key);
  if (!match) return;
  const parts = match[1].replace(/datasets\[x\]/g, 'datasets[0]').split('.');
  let node = process;
  for (let index = 0; index < parts.length; index += 1) {
    const indexed = /^(\w+)\[(\d+)\]$/.exec(parts[index]);
    const last = index === parts.length - 1;
    if (indexed) {
      node = node[indexed[1]][Number(indexed[2])];
      if (last) throw new Error(`cannot assign an array element for ${key}`);
      continue;
    }
    if (last) node[parts[index]] = value === undefined ? undefined : JSON.parse(JSON.stringify(value));
    else node = node[parts[index]] = node[parts[index]] ?? {};
  }
}

function pick(section, keys) {
  const out = {};
  for (const key of keys) if (section && section[key] !== undefined) out[key] = section[key];
  return out;
}

const baseProcess = jobConfigModule.defaultJobConfig.config.process[0];
const startArch = baseProcess.model.arch;
const start = archs.get(startArch);
if (!start) throw new Error(`UI base arch ${startArch} has no entry`);

const entries = {};
for (const [name, entry] of [...archs.entries()].sort()) {
  const process = JSON.parse(JSON.stringify(baseProcess));
  if (name !== startArch) {
    if (!(entry.additionalSections || []).includes('model.low_vram')) process.model.low_vram = false;
    for (const [key, pair] of Object.entries(start.defaults || {})) setPath(process, key, pair[1]);
    for (const [key, pair] of Object.entries(entry.defaults || {})) setPath(process, key, pair[0]);
  }
  const nameOrPath = (entry.defaults || {})['config.process[0].model.name_or_path'];
  entries[name] = {
    arch: name.split(':')[0] === 'flex1' ? 'flux' : name.split(':')[0],
    name_or_path: Array.isArray(nameOrPath) ? nameOrPath[0] : null,
    train: pick(process.train, TRAIN_KEYS),
    model: pick(process.model, MODEL_KEYS),
    dataset: pick(process.datasets[0], DATASET_KEYS),
  };
}

const commit = require('child_process').execSync('git -C /app/ai-toolkit rev-parse HEAD').toString().trim();
process.stdout.write(JSON.stringify({ commit, sources, base_arch: startArch, entries }));
