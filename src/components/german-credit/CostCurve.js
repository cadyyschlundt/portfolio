"use client"; // Marks this component to run in the browser, not just render on the server - Recharts needs the DOM for hover interactivity.

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

export default function CostCurve({ points }) {
  return (
    // ResponsiveContainer: measures its parent's width and resizes the chart to fit it; a fixed height keeps the page from jumping as data loads.
    <ResponsiveContainer width="100%" height={400}>
      {/* LineChart: the chart itself - takes the points array and lays out a shared x/y coordinate space for its children below. */}
      <LineChart data={points} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
        {/* XAxis: reads "threshold" off each point and draws the horizontal axis, labeled so a reader knows what it means. */}
        <XAxis
          dataKey="threshold"
          label={{ value: "Decision threshold", position: "insideBottom", offset: -10 }}
        />
        {/* YAxis: reads "cost" off each point and draws the vertical axis, labeled the same way. */}
        <YAxis
          label={{ value: "Average cost per applicant", angle: -90, position: "insideLeft" }}
        />
        {/* Tooltip: shows the exact threshold/cost pair for whatever point the mouse is nearest to - this is the "hover" part that requires the client. */}
        <Tooltip />
        {/* Line: the actual curve - plots cost (y) against threshold (x) for every point in the data array. dot={false} keeps 101 points from turning into 101 markers. */}
        <Line type="monotone" dataKey="cost" dot={false} />
        {/* ReferenceLine (x=0.20): a vertical line marking the chosen threshold, so it's visible against the curve, not just implied by a tooltip. */}
        <ReferenceLine x={0.2} label="chosen (0.20)" stroke="green" />
        {/* ReferenceLine (x=0.50): a vertical line marking the default threshold, for the same reason - the two lines together are the point of the chart. */}
        <ReferenceLine x={0.5} label="default (0.50)" stroke="red" />
      </LineChart>
    </ResponsiveContainer>
  );
}
