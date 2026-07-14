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

module.exports = { classify };
