"use client";

import { useEffect, useRef, useState } from "react";
import { classify } from "@/lib/knn";

const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 400;
const POINT_RADIUS = 5;
const CELL_SIZE = 8;
const BOUNDARY_ALPHA = 0.25;
const DEFAULT_K = 5;
const MIN_K = 1;
const MAX_K = 25;

const CLASS_COLORS = {
  A: "#2563eb",
  B: "#dc2626",
};

function withAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export default function PointCanvas() {
  const canvasRef = useRef(null);
  const [points, setPoints] = useState([]);
  const [activeClass, setActiveClass] = useState("A");
  const [k, setK] = useState(DEFAULT_K);

  useEffect(() => {
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = CANVAS_WIDTH * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    for (let gx = 0; gx < CANVAS_WIDTH; gx += CELL_SIZE) {
      for (let gy = 0; gy < CANVAS_HEIGHT; gy += CELL_SIZE) {
        const label = classify(
          { x: gx + CELL_SIZE / 2, y: gy + CELL_SIZE / 2 },
          points,
          k
        );
        if (label === null) continue;
        ctx.fillStyle = withAlpha(CLASS_COLORS[label], BOUNDARY_ALPHA);
        ctx.fillRect(gx, gy, CELL_SIZE, CELL_SIZE);
      }
    }

    for (const point of points) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, POINT_RADIUS, 0, 2 * Math.PI);
      ctx.fillStyle = CLASS_COLORS[point.label];
      ctx.fill();
    }
  }, [points, k]);

  function handleCanvasClick(event) {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    setPoints((prev) => [...prev, { x, y, label: activeClass }]);
  }

  const countA = points.filter((p) => p.label === "A").length;
  const countB = points.filter((p) => p.label === "B").length;
  const effectiveK = Math.min(k, points.length);

  return (
    <div>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setActiveClass("A")}
          className={`rounded px-3 py-1 border ${
            activeClass === "A"
              ? "bg-blue-600 text-white border-blue-600"
              : "border-zinc-300 dark:border-zinc-700"
          }`}
        >
          Class A
        </button>
        <button
          type="button"
          onClick={() => setActiveClass("B")}
          className={`rounded px-3 py-1 border ${
            activeClass === "B"
              ? "bg-red-600 text-white border-red-600"
              : "border-zinc-300 dark:border-zinc-700"
          }`}
        >
          Class B
        </button>
        <button
          type="button"
          onClick={() => setPoints([])}
          className="rounded px-3 py-1 border border-zinc-300 dark:border-zinc-700"
        >
          Clear
        </button>
      </div>

      <div className="mt-3 flex items-center gap-2">
        <label htmlFor="k-slider" className="text-sm">
          k: {k}
        </label>
        <input
          id="k-slider"
          type="range"
          min={MIN_K}
          max={MAX_K}
          value={k}
          onChange={(event) => setK(Number(event.target.value))}
        />
      </div>

      <div className="mt-3 inline-block border border-zinc-400 dark:border-zinc-600">
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="block w-[600px] h-[400px]"
        />
      </div>

      <p className="mt-2 text-sm">
        Class A: {countA} &nbsp; Class B: {countB} &nbsp; Effective k:{" "}
        {effectiveK}
      </p>
    </div>
  );
}
