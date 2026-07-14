function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function classify(point, points, k) {
  if (points.length === 0) return null;

  const neighbors = points
    .map((p) => ({ label: p.label, distance: distance(point, p) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, Math.min(k, points.length));

  const votes = new Map();
  for (const neighbor of neighbors) {
    votes.set(neighbor.label, (votes.get(neighbor.label) || 0) + 1);
  }

  let maxVotes = 0;
  for (const count of votes.values()) {
    if (count > maxVotes) maxVotes = count;
  }

  const tiedLabels = [...votes.keys()].filter(
    (label) => votes.get(label) === maxVotes
  );
  if (tiedLabels.length === 1) return tiedLabels[0];

  for (const neighbor of neighbors) {
    if (tiedLabels.includes(neighbor.label)) return neighbor.label;
  }

  return null;
}

function majorityBaseline(points) {
  if (points.length === 0) return null;

  const counts = new Map();
  for (const point of points) {
    counts.set(point.label, (counts.get(point.label) || 0) + 1);
  }

  let maxCount = 0;
  for (const count of counts.values()) {
    if (count > maxCount) maxCount = count;
  }

  return maxCount / points.length;
}

function loocvError(points, k) {
  if (points.length < 2) return null;

  let misclassified = 0;
  for (let i = 0; i < points.length; i++) {
    const query = points[i];
    const others = points.slice(0, i).concat(points.slice(i + 1));
    const predicted = classify(query, others, k);
    if (predicted !== query.label) misclassified++;
  }

  return misclassified / points.length;
}

module.exports = { classify, majorityBaseline, loocvError };
