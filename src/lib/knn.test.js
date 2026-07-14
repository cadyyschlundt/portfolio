const assert = require("assert");
const { classify, majorityBaseline, loocvError } = require("./knn.js");

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

// 6. majorityBaseline on an empty array returns null.
assert.strictEqual(
  majorityBaseline([]),
  null,
  "majorityBaseline of no points should return null"
);

// 7. majorityBaseline is the fraction held by the most common label.
const skewedPoints = [
  { x: 0, y: 0, label: "A" },
  { x: 0, y: 1, label: "A" },
  { x: 0, y: 2, label: "A" },
  { x: 10, y: 10, label: "B" },
];
assert.strictEqual(
  majorityBaseline(skewedPoints),
  0.75,
  "majorityBaseline should be the majority class's share of all points"
);

// 8. loocvError needs at least 2 points.
assert.strictEqual(
  loocvError([], 3),
  null,
  "loocvError of no points should return null"
);
assert.strictEqual(
  loocvError([{ x: 0, y: 0, label: "A" }], 3),
  null,
  "loocvError of a single point should return null"
);

// 9. At k = n - 1, LOOCV degenerates to the majority-vote baseline: the
// held-out B point always loses to the remaining 3 A's.
assert.strictEqual(
  loocvError(skewedPoints, 3),
  0.25,
  "LOOCV error at k = n - 1 should match the majority baseline error"
);

// 10. Two well-separated clusters: at k = 1, every held-out point's
// nearest remaining neighbor is still in its own cluster.
const separatedClusters = [
  { x: 0, y: 0, label: "A" },
  { x: 0, y: 1, label: "A" },
  { x: 10, y: 10, label: "B" },
  { x: 10, y: 11, label: "B" },
];
assert.strictEqual(
  loocvError(separatedClusters, 1),
  0,
  "well-separated clusters should have zero LOOCV error at k = 1"
);

console.log("All knn.js assertions passed.");
