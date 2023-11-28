import { env, pipeline } from "@xenova/transformers";

env.localModelPath = 'node_modules/embedding/model';
let pipe: Function | null = null;
const piping = pipeline("feature-extraction", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", { local_files_only: true, quantized: true }).then(p => { pipe = p; })

export async function embedding_calc(text: string) {
    await piping;
    if (pipe) {
        return await pipe(text);
    }
}