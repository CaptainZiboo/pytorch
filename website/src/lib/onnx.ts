import * as ort from "onnxruntime-web";

export interface Prediction {
  digit: number;
  confidence: number;
  probs: number[];
}

let session: ort.InferenceSession | null = null;

export async function load(path = "/pytorch/model.onnx") {
  if (session) return;

  try {
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/";
    session = await ort.InferenceSession.create(path);
  } catch (error) {
    console.error("Erreur lors du chargement du modèle ONNX:", error);
    throw new Error(`Impossible de charger le modèle: ${error}`);
  }
}

export async function predict(input: Float32Array): Promise<Prediction> {
  if (!session) {
    throw new Error(
      "Le modèle ONNX n'est pas encore chargé. Appelez load() d'abord."
    );
  }

  if (input.length !== 784) {
    throw new Error(
      `Input invalide: attendu 784 valeurs, reçu ${input.length}`
    );
  }

  try {
    const tensor = new ort.Tensor("float32", input, [1, 1, 28, 28]);
    const feeds: Record<string, ort.Tensor> = {
      [session.inputNames[0]]: tensor,
    };

    const output = await session.run(feeds);
    const outputTensor = output[session.outputNames[0]];

    if (!outputTensor || !outputTensor.data) {
      throw new Error("Sortie du modèle invalide");
    }

    const logits = Array.from(outputTensor.data as Float32Array);

    if (logits.length !== 10) {
      throw new Error(
        `Sortie inattendue: attendu 10 classes, reçu ${logits.length}`
      );
    }

    const max = Math.max(...logits);
    const exps = logits.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);

    if (sum === 0) {
      throw new Error("Erreur de softmax: somme nulle");
    }

    const probs = exps.map((x) => x / sum);

    const maxProb = Math.max(...probs);
    const maxIdx = probs.findIndex((v) => v === maxProb);

    return {
      digit: maxIdx,
      confidence: maxProb * 100,
      probs,
    };
  } catch (error) {
    console.error("Erreur lors de la prédiction:", error);
    throw new Error(`Prédiction échouée: ${error}`);
  }
}

export async function dispose() {
  if (session) {
    await session.release();
    session = null;
  }
}
