import React, { useRef, useEffect, useState, useCallback, useMemo } from "react";
import { load, predict, type Prediction } from "./lib/onnx";

interface FullPrediction extends Prediction {
  probs: number[];
}

const Canvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const debugCanvasRef = useRef<HTMLCanvasElement>(null);
  const [result, setResult] = useState<FullPrediction | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);

  const tempCanvases = useMemo(() => ({
    crop: document.createElement("canvas"),
    final: document.createElement("canvas"),
  }), []);

  const setupCanvas = useCallback((canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 20;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) setupCanvas(canvas);
  }, [setupCanvas]);

  useEffect(() => {
    const loadModel = async () => {
      try {
        await load();
        setModelLoaded(true);
        console.log("✅ Modèle ONNX chargé avec succès");
      } catch (error) {
        console.error("❌ Erreur lors du chargement du modèle:", error);
      }
    };
    loadModel();
  }, []);

  const findContentBounds = useCallback((imageData: ImageData) => {
    const { data, width, height } = imageData;
    let minX = width, minY = height, maxX = 0, maxY = 0;
    let hasContent = false;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        const alpha = data[idx + 3];

        if (alpha > 0 && (r < 250 || g < 250 || b < 250)) {
          hasContent = true;
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }

    return { minX, minY, maxX, maxY, hasContent };
  }, []);

  const cropImage = useCallback((
    sourceCanvas: HTMLCanvasElement,
    bounds: { minX: number; minY: number; maxX: number; maxY: number }
  ) => {
    const { minX, minY, maxX, maxY } = bounds;
    const cropWidth = maxX - minX + 1;
    const cropHeight = maxY - minY + 1;
    const cropSize = Math.max(cropWidth, cropHeight);
    const padding = Math.max(20, cropSize * 0.1);
    const expandedCropSize = cropSize + 2 * padding;

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const cropStartX = Math.round(centerX - expandedCropSize / 2);
    const cropStartY = Math.round(centerY - expandedCropSize / 2);

    // Réutiliser le canvas temporaire
    const cropCanvas = tempCanvases.crop;
    cropCanvas.width = expandedCropSize;
    cropCanvas.height = expandedCropSize;
    const cropCtx = cropCanvas.getContext("2d")!;

    cropCtx.fillStyle = "#fff";
    cropCtx.fillRect(0, 0, expandedCropSize, expandedCropSize);
    cropCtx.drawImage(
      sourceCanvas,
      cropStartX, cropStartY, expandedCropSize, expandedCropSize,
      0, 0, expandedCropSize, expandedCropSize
    );

    return { cropCanvas, expandedCropSize };
  }, [tempCanvases.crop]);

  const resizeTo28x28 = useCallback((
    cropCanvas: HTMLCanvasElement,
    cropSize: number
  ) => {
    const finalCanvas = tempCanvases.final;
    finalCanvas.width = 28;
    finalCanvas.height = 28;
    const finalCtx = finalCanvas.getContext("2d")!;

    finalCtx.imageSmoothingEnabled = true;
    finalCtx.imageSmoothingQuality = "high";
    finalCtx.fillStyle = "#fff";
    finalCtx.fillRect(0, 0, 28, 28);

    const margin = 2;
    const targetSize = 28 - 2 * margin;
    finalCtx.drawImage(
      cropCanvas,
      0, 0, cropSize, cropSize,
      margin, margin, targetSize, targetSize
    );

    return finalCanvas;
  }, [tempCanvases.final]);

  const applyMorphologicalProcessing = useCallback((canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext("2d")!;
    const imageData = ctx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    const originalData = new Uint8ClampedArray(data);

    for (let y = 1; y < 27; y++) {
      for (let x = 1; x < 27; x++) {
        const idx = (y * 28 + x) * 4;
        const gray = 0.299 * originalData[idx] + 0.587 * originalData[idx + 1] + 0.114 * originalData[idx + 2];

        if (gray < 200) {
          const neighbors = [
            [-1, 0], [1, 0], [0, -1], [0, 1]
          ];

          for (const [dx, dy] of neighbors) {
            const neighborIdx = ((y + dy) * 28 + (x + dx)) * 4;
            const neighborGray = 0.299 * originalData[neighborIdx] + 0.587 * originalData[neighborIdx + 1] + 0.114 * originalData[neighborIdx + 2];

            if (neighborGray > 230) {
              data[neighborIdx] = Math.max(0, data[neighborIdx] - 30);
              data[neighborIdx + 1] = Math.max(0, data[neighborIdx + 1] - 30);
              data[neighborIdx + 2] = Math.max(0, data[neighborIdx + 2] - 30);
            }
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, []);

  const imageToTensor = useCallback((canvas: HTMLCanvasElement): Float32Array => {
    const ctx = canvas.getContext("2d")!;
    const imageData = ctx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    const tensor = new Float32Array(784);

    for (let i = 0; i < 784; i++) {
      const pixelIdx = i * 4;
      const r = data[pixelIdx];
      const g = data[pixelIdx + 1];
      const b = data[pixelIdx + 2];

      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      let normalized = (gray / 255.0 - 0.5) / 0.5;
      normalized = Math.tanh(normalized * 2) * 0.8;
      tensor[i] = -normalized;
    }

    return tensor;
  }, []);

  const processCanvasToTensor = useCallback((canvas: HTMLCanvasElement): Float32Array => {
    const ctx = canvas.getContext("2d")!;
    const originalImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    const bounds = findContentBounds(originalImageData);
    if (!bounds.hasContent) {
      const emptyTensor = new Float32Array(784);
      visualizeTensor(emptyTensor.map(x => (x + 1) / 2));
      return emptyTensor;
    }

    const { cropCanvas, expandedCropSize } = cropImage(canvas, bounds);

    const finalCanvas = resizeTo28x28(cropCanvas, expandedCropSize);

    applyMorphologicalProcessing(finalCanvas);

    const tensor = imageToTensor(finalCanvas);

    visualizeTensor(tensor.map(x => Math.max(0, Math.min(1, (x + 1) / 2))));

    return tensor;
  }, [findContentBounds, cropImage, resizeTo28x28, applyMorphologicalProcessing, imageToTensor]);

  const visualizeTensor = useCallback((tensor: Float32Array) => {
    const debugCanvas = debugCanvasRef.current;
    if (!debugCanvas) return;

    const ctx = debugCanvas.getContext("2d")!;
    debugCanvas.width = 224;
    debugCanvas.height = 224;

    const nonZeroCount = tensor.filter(x => Math.abs(x) > 0.01).length;
    console.log("📊 Tensor - Pixels non-zéro:", nonZeroCount);

    const imageData = ctx.createImageData(224, 224);
    const data = imageData.data;

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const tensorIdx = y * 28 + x;
        const value = Math.max(0, Math.min(1, tensor[tensorIdx]));
        const grayValue = Math.round(value * 255);

        for (let dy = 0; dy < 8; dy++) {
          for (let dx = 0; dx < 8; dx++) {
            const pixelIdx = ((y * 8 + dy) * 224 + (x * 8 + dx)) * 4;
            data[pixelIdx] = grayValue;     // R
            data[pixelIdx + 1] = grayValue; // G
            data[pixelIdx + 2] = grayValue; // B
            data[pixelIdx + 3] = 255;       // A
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);

    ctx.strokeStyle = "#ff000020";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 28; i += 7) { // Grille moins dense
      ctx.beginPath();
      ctx.moveTo(i * 8, 0);
      ctx.lineTo(i * 8, 224);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * 8);
      ctx.lineTo(224, i * 8);
      ctx.stroke();
    }
  }, []);

  const getPointerPosition = useCallback((
    e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>
  ) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    let clientX: number, clientY: number;

    if ('touches' in e) {
      if (e.touches.length === 0) return null;
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  }, []);

  const startDraw = useCallback((
    e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>
  ) => {
    if (!modelLoaded) return;
    e.preventDefault();

    const pos = getPointerPosition(e);
    if (!pos) return;

    setDrawing(true);
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx) return;

    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }, [modelLoaded, getPointerPosition]);

  const draw = useCallback((
    e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>
  ) => {
    if (!drawing || !modelLoaded) return;
    e.preventDefault();

    const pos = getPointerPosition(e);
    if (!pos) return;

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx) return;

    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  }, [drawing, modelLoaded, getPointerPosition]);

  const endDraw = useCallback(async () => {
    if (!drawing || !modelLoaded) return;

    setDrawing(false);
    setIsLoading(true);

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (ctx) ctx.beginPath();

    try {
      const inputTensor = processCanvasToTensor(canvas);
      const prediction = await predict(inputTensor);
      setResult(prediction as FullPrediction);
      console.log("🎯 Prédiction:", prediction);
    } catch (error) {
      console.error("❌ Erreur de prédiction:", error);
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  }, [drawing, modelLoaded, processCanvasToTensor]);

  const resetCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    setupCanvas(canvas);

    const debugCanvas = debugCanvasRef.current;
    if (debugCanvas) {
      const debugCtx = debugCanvas.getContext("2d");
      if (debugCtx) {
        debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
        debugCtx.fillStyle = "#f0f0f0";
        debugCtx.fillRect(0, 0, debugCanvas.width, debugCanvas.height);
      }
    }

    setResult(null);
    setDrawing(false);
  }, [setupCanvas]);

  const probsToShow = result?.probs ?? Array(10).fill(0);
  const predictedDigit = result?.digit ?? -1;

  return (
    <div className="flex w-screen h-screen overflow-hidden p-4 gap-4">
      <div className="flex flex-col w-2/3 h-full">
        <div className="flex-1 flex items-center justify-center bg-gray-100 rounded-lg p-4 relative min-h-0">
          <div className="w-full h-full max-w-[450px] max-h-[450px] aspect-square relative">
            <canvas
              ref={canvasRef}
              width={450}
              height={450}
              className={`w-full h-full border-2 border-gray-300 bg-white rounded-lg shadow-lg transition-all touch-none ${!modelLoaded
                ? "cursor-not-allowed opacity-50"
                : drawing
                  ? "cursor-crosshair border-blue-400"
                  : "cursor-crosshair hover:border-gray-400"
                }`}
              onMouseDown={startDraw}
              onMouseMove={draw}
              onMouseUp={endDraw}
              onMouseLeave={endDraw}
              onTouchStart={startDraw}
              onTouchMove={draw}
              onTouchEnd={endDraw}
            />

            {(isLoading || !modelLoaded) && (
              <div className="absolute inset-0 bg-black bg-opacity-20 flex items-center justify-center rounded-lg">
                <div className="bg-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent"></div>
                  <span className="text-sm font-medium">
                    {!modelLoaded ? "Chargement du modèle..." : "Prédiction en cours..."}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="h-16 flex items-center gap-4 px-2 flex-shrink-0">
          <button
            className={`px-6 py-2 rounded-lg font-medium transition-all ${!modelLoaded
              ? "bg-gray-300 text-gray-500 cursor-not-allowed"
              : "bg-red-600 text-white hover:bg-red-700 hover:shadow-md"
              }`}
            onClick={resetCanvas}
            disabled={!modelLoaded}
          >
            🗑️ Reset
          </button>

          <div className="text-sm text-gray-600 flex-1">
            {!modelLoaded ? (
              "⏳ Chargement..."
            ) : (
              "✅ Modèle prêt - Dessinez un chiffre !"
            )}
          </div>
        </div>
      </div>

      <div className="flex flex-col flex-1 min-w-0 p-4 bg-gray-50 rounded-lg overflow-y-auto">
        <h3 className="text-xl font-bold mb-4 text-gray-800">🧠 Résultat</h3>

        <div className="flex gap-4 items-center mb-4">
          <div className="flex flex-col w-[250px] bg-gray-50 rounded-lg flex-shrink-0">
            <div className="bg-white p-3 rounded-lg shadow-sm flex flex-col items-center flex-1">
              <canvas
                ref={debugCanvasRef}
                width={224}
                height={224}
                className="border border-gray-400 rounded max-w-full max-h-full"
                style={{ imageRendering: 'pixelated' }}
              />
              <p className="text-xs text-gray-500 mt-2 text-center">
                28×28 pixels<br />
                Noir = 1.0, Blanc = 0.0
              </p>
            </div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm flex-shrink-0 flex-1 h-full flex items-center justify-center">
            <div className="text-center">
              <div className="text-5xl font-bold mb-2 text-blue-600">
                {predictedDigit === -1 ? "?" : predictedDigit}
              </div>
              <p className="text-base text-gray-600">
                Confiance: {" "}
                <span className="font-semibold text-gray-800">
                  {predictedDigit === -1 ? "0%" : `${result!.confidence.toFixed(1)}%`}
                </span>
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-sm flex-1 min-h-0 overflow-y-auto">
          <h4 className="font-semibold mb-3 text-gray-700">Probabilités</h4>
          <div className="space-y-2">
            {probsToShow.map((prob, i) => (
              <div key={i} className="group">
                <div className="flex justify-between text-sm mb-1">
                  <span className={`font-medium ${i === predictedDigit ? 'text-green-600' : 'text-gray-600'}`}>
                    {i}
                  </span>
                  <span className={`${i === predictedDigit ? 'text-green-600 font-semibold' : 'text-gray-500'}`}>
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-2 rounded-full transition-all duration-500 ease-out ${i === predictedDigit
                      ? 'bg-gradient-to-r from-green-400 to-green-600'
                      : 'bg-gray-400'
                      }`}
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Canvas;