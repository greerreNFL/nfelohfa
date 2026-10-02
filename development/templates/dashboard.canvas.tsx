import {
  Callout,
  Divider,
  H1,
  H2,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

/**
 * Copy this file. Fill WEIGHT_*, LIFT_*, stats, table rows, and the callout
 * from the 40k run and FINDINGS.md. Keep 0 at the center of both charts.
 */

const WEIGHT_X = [
  -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3,
  0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
];
const WEIGHT_Y = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

const LIFT_X = [
  -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05,
  0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
];
const LIFT_ADJ = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const LIFT_BASE = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

type DistSeries = {
  name: string;
  data: number[];
  role: "primary" | "muted";
};

function DistArea({
  x,
  series,
  xLabel,
  yLabel,
  height = 220,
  tickEvery = 2,
  xFormat,
}: {
  x: number[];
  series: DistSeries[];
  xLabel: string;
  yLabel: string;
  height?: number;
  tickEvery?: number;
  xFormat?: (v: number) => string;
}) {
  const theme = useHostTheme();
  const primary = theme.accent.primary;
  const muted = theme.text.quaternary;
  const axis = theme.text.tertiary;
  const grid = theme.stroke.tertiary;
  const zero = theme.text.secondary;
  const fmt = xFormat ?? ((v: number) => String(v));
  const padL = 48;
  const padR = 12;
  const padT = 14;
  const padB = 44;
  const w = 720;
  const innerW = w - padL - padR;
  const innerH = height - padT - padB;
  const xMin = x[0];
  const xMax = x[x.length - 1];
  const yMax = Math.max(...series.flatMap((s) => s.data), 1);
  const xAt = (v: number) => padL + ((v - xMin) / (xMax - xMin)) * innerW;
  const yAt = (v: number) => padT + innerH - (v / yMax) * innerH;
  const zeroX = xAt(0);

  function linePath(data: number[]) {
    return data
      .map((v, i) => `${i === 0 ? "M" : "L"} ${xAt(x[i]).toFixed(2)} ${yAt(v).toFixed(2)}`)
      .join(" ");
  }
  function areaPath(data: number[]) {
    const top = linePath(data);
    const lastX = xAt(x[x.length - 1]).toFixed(2);
    const firstX = xAt(x[0]).toFixed(2);
    const base = yAt(0).toFixed(2);
    return `${top} L ${lastX} ${base} L ${firstX} ${base} Z`;
  }

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${w} ${height}`}
      role="img"
      style={{ display: "block" }}
    >
      <line
        x1={zeroX}
        y1={padT}
        x2={zeroX}
        y2={padT + innerH}
        stroke={zero}
        strokeWidth={1.5}
      />
      <text
        x={zeroX + 6}
        y={padT + 12}
        fill={zero}
        fontSize={11}
        fontFamily="inherit"
      >
        0
      </text>
      {series.map((s) => {
        const color = s.role === "primary" ? primary : muted;
        const width = s.role === "primary" ? 2.5 : 1.75;
        return (
          <g key={s.name}>
            <path d={areaPath(s.data)} fill={color} fillOpacity={0.16} />
            <path
              d={linePath(s.data)}
              fill="none"
              stroke={color}
              strokeWidth={width}
            />
          </g>
        );
      })}
      <line
        x1={padL}
        y1={padT + innerH}
        x2={padL + innerW}
        y2={padT + innerH}
        stroke={grid}
      />
      {x.map((v, i) =>
        i % tickEvery === 0 ? (
          <text
            key={v}
            x={xAt(v)}
            y={height - 22}
            textAnchor="middle"
            fill={axis}
            fontSize={10}
            fontFamily="inherit"
          >
            {fmt(v)}
          </text>
        ) : null
      )}
      <text
        x={padL + innerW / 2}
        y={height - 4}
        textAnchor="middle"
        fill={axis}
        fontSize={11}
        fontFamily="inherit"
      >
        {xLabel}
      </text>
      <text
        x={12}
        y={padT + innerH / 2}
        textAnchor="middle"
        fill={axis}
        fontSize={11}
        fontFamily="inherit"
        transform={`rotate(-90 12 ${padT + innerH / 2})`}
      >
        {yLabel}
      </text>
    </svg>
  );
}

export default function FeatureDashboard() {
  const theme = useHostTheme();

  return (
    <Stack gap={22}>
      <Stack gap={6}>
        <H1>{"<feature>"}</H1>
        <Text tone="secondary">
          {"<one-line definition. sample filters.>"}
        </Text>
      </Stack>

      <Callout tone="neutral" title="<ship or do not ship, weight if ship>">
        {"<four legs: research, residual, mechanics, 40k.>"}
      </Callout>

      <Row justify="space-between" style={{ width: "100%" }}>
        <Stat value="—" label="Median weight" />
        <Stat value="—" label="Draws below 0" />
        <Stat value="—" label="5th–95th" />
        <Stat value="—" label="Median × base" />
      </Row>

      <Divider />

      <Stack gap={6}>
        <H2>{"<feature> weight"}</H2>
        <Text tone="tertiary">
          X: coefficient (fraction of hfa_base), 0.10-wide bins. Y: runs.
          In-model draws only. Vertical line is 0.
        </Text>
        <DistArea
          x={WEIGHT_X}
          series={[{ name: "<feature>", data: WEIGHT_Y, role: "primary" }]}
          xLabel="<feature> weight"
          yLabel="Runs"
          tickEvery={2}
          xFormat={(v) => (v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1))}
        />
      </Stack>

      <Stack gap={8}>
        <H2>Features</H2>
        <Text tone="tertiary">
          Conditional lift: RMSE on feature-on games, with vs without that
          adjustment. Total lift: same, on all games. Straddle rate is the
          mass on the minority side of 0. Mag. ratio is
          min(|even|,|odd|) / max(|even|,|odd|) of the on-minus-off residual.
        </Text>
        <Table
          headers={[
            "Feature",
            "% of games",
            "Conditional lift",
            "Total lift",
            "Median",
            "p05",
            "p95",
            "% < 0",
            "Straddle rate",
            "Mag. ratio",
          ]}
          columnAlign={[
            "left",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
          ]}
          striped
          rows={[
            [
              "home_bye",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
            ],
            [
              "away_bye",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
            ],
            [
              "home_time_advantage",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
            ],
            [
              "dif_surface",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
            ],
            [
              "div_game",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
            ],
            [
              "<feature>",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
              "—",
            ],
          ]}
        />
      </Stack>

      <Divider />

      <Stack gap={6}>
        <H2>Test-set RMSE lift vs flat 2.5</H2>
        <Text tone="tertiary">
          Lift = static_RMSE / model_RMSE − 1, in percent. X centered at 0.
        </Text>
        <Row gap={16} align="center">
          <span
            style={{
              width: 10,
              height: 10,
              background: theme.accent.primary,
              display: "inline-block",
            }}
          />
          <Text size="small">Adjusted</Text>
          <span
            style={{
              width: 10,
              height: 10,
              background: theme.text.quaternary,
              display: "inline-block",
            }}
          />
          <Text size="small" tone="tertiary">
            Base only (rolling HFA)
          </Text>
        </Row>
        <DistArea
          x={LIFT_X}
          series={[
            { name: "Adjusted", data: LIFT_ADJ, role: "primary" },
            { name: "Base only (rolling HFA)", data: LIFT_BASE, role: "muted" },
          ]}
          xLabel="Test lift (%)"
          yLabel="Runs"
          tickEvery={2}
          xFormat={(v) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
        />
      </Stack>

      <Row justify="space-between" style={{ width: "100%" }}>
        <Stat value="—" label="Mean adj lift" />
        <Stat value="—" label="Mean base lift" />
        <Stat value="—" label="Complete-feat lift" />
        <Stat value="—" label="Hold out new feature" />
      </Row>

    </Stack>
  );
}
