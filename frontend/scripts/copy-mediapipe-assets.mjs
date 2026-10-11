import { cp, mkdir, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const wasmSource = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const wasmDestination = path.join(root, 'public', 'mediapipe', 'wasm');
const model = path.join(root, 'public', 'models', 'hand_landmarker.task');

try {
  await stat(model);
} catch {
  console.error('\nFalta frontend/public/models/hand_landmarker.task.');
  console.error('Copia allí el archivo hand_landmarker.task de tu proyecto local antes de construir.\n');
  process.exit(1);
}
await rm(wasmDestination, { recursive: true, force: true });
await mkdir(path.dirname(wasmDestination), { recursive: true });
await cp(wasmSource, wasmDestination, { recursive: true });
console.log('MediaPipe WASM copiado a public/mediapipe/wasm.');
