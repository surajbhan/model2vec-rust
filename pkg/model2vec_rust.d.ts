/* tslint:disable */
/* eslint-disable */

export class Model2VecWasm {
    free(): void;
    [Symbol.dispose](): void;
    dim(): number;
    encode(text: string, add_special_tokens: boolean): Float32Array;
    encode_batch(sentences: any[], add_special_tokens: boolean): Float32Array;
    constructor(config_bytes: Uint8Array, tokenizer_bytes: Uint8Array, safetensors_bytes: Uint8Array);
    vocab_size(): number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_model2vecwasm_free: (a: number, b: number) => void;
    readonly model2vecwasm_dim: (a: number) => number;
    readonly model2vecwasm_encode: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly model2vecwasm_encode_batch: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly model2vecwasm_new: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly model2vecwasm_vocab_size: (a: number) => number;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
