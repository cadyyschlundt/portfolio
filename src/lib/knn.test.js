const assert = require("assert");
const { classify } = require("./knn.js");

// 1. Empty points array returns null, not an error and not a string.
assert.strictEqual(
  classify({ x: 0, y: 0 }, [], 3),
  null,
  "empty points should return null"
);

// 2. Single point: the only label available wins regardless of k.
const singlePoint = [{ x: 5, y: 5, label: "A" }];
assert.strictEqual(
  classify({ x: 0, y: 0 }, singlePoint, 1),
  "A",
  "single point should return its own label"
);

// 3. k greater than the number of points clamps to all points.
const fewPoints = [
  { x: 0, y: 0, label: "A" },
  { x: 1, y: 0, label: "A" },
  { x: 2, y: 0, label: "B" },
];
assert.strictEqual(
  classify({ x: 0, y: 0 }, fewPoints, 10),
  "A",
  "k > n should clamp to n and return the overall majority"
);

// 4. An obvious tie: two classes split the vote, so the class with the
// closer of the two tied points should win.
const tiedPoints = [
  { x: 1, y: 0, label: "A" }, // distance 1
  { x: 2, y: 0, label: "B" }, // distance 2
];
assert.strictEqual(
  classify({ x: 0, y: 0 }, tiedPoints, 2),
  "A",
  "a tie should be broken by whichever tied class has the nearer neighbor"
);

// 5. A clear-cut case: three points from the same class dominate the
// vote at k=3.
const clearCutPoints = [
  { x: 0, y: 0, label: "A" },
  { x: 0, y: 1, label: "A" },
  { x: 0, y: 2, label: "A" },
  { x: 10, y: 10, label: "B" },
];
assert.strictEqual(
  classify({ x: 0, y: 0.5 }, clearCutPoints, 3),
  "A",
  "a clear majority among the k nearest should win"
);

console.log("All knn.js assertions passed.");
