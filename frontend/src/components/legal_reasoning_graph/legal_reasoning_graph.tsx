import { useMemo } from "react";

import {
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
} from "@xyflow/react";

import type { Edge, Node, NodeProps } from "@xyflow/react";
import type { ReasoningMap, ReasoningNode } from "../../api";

import "@xyflow/react/dist/style.css";
import "./legal_reasoning_graph.css";

type LegalNodeKind =
  | "issue"
  | "condition"
  | "test"
  | "outcome"
  | "consequence"
  | "related";

type LegalNodeData = {
  kind: LegalNodeKind;
  stepLabel: string;
  title: string;
  sourceLabel?: string;
  sourceUrl?: string;
};

type LegalFlowNode = Node<LegalNodeData, "legalNode">;

function getDisplayStepLabel(node: ReasoningNode): string {
  const nodeType = node.node_type.toLowerCase();

  if (nodeType === "issue") {
    return "Legal issue";
  }

  if (nodeType === "condition") {
    return "Condition";
  }

  if (nodeType === "test" || nodeType === "legal_test") {
    return "How to decide";
  }

  if (nodeType === "outcome") {
    return "Outcome";
  }

  if (nodeType === "consequence" || nodeType === "remedy") {
    return "Consequence";
  }

  if (
    nodeType === "additional_check" ||
    nodeType === "related" ||
    nodeType === "exception" ||
    nodeType === "limitation"
  ) {
    return "Related point";
  }

  return node.node_type.replaceAll("_", " ");
}

function getNodeKind(node: ReasoningNode): LegalNodeKind {
  const nodeType = node.node_type.toLowerCase();

  if (nodeType === "issue") {
    return "issue";
  }

  if (nodeType === "condition") {
    return "condition";
  }

  if (nodeType === "test" || nodeType === "legal_test") {
    return "test";
  }

  if (nodeType === "outcome") {
    return "outcome";
  }

  if (nodeType === "consequence" || nodeType === "remedy") {
    return "consequence";
  }

  return "related";
}

function getSourceLabel(node: ReasoningNode): string | undefined {
  if (node.article_label) {
    return node.article_label;
  }

  return undefined;
}

function getNodeTitle(node: ReasoningNode): string {
  return node.label.trim();
}

function LegalNode({ data }: NodeProps<LegalFlowNode>) {
  return (
    <div className={`legal-flow-node legal-flow-node--${data.kind}`}>
      <Handle type="target" position={Position.Left} className="legal-handle" />

      <div className="legal-node-label">{data.stepLabel}</div>
      <div className="legal-node-title">{data.title}</div>

      {data.sourceLabel && data.sourceUrl && (
        <a
          href={data.sourceUrl}
          target="_blank"
          rel="noreferrer"
          className="legal-source-chip"
          onClick={(event) => event.stopPropagation()}
        >
          {data.sourceLabel}
        </a>
      )}

      {data.sourceLabel && !data.sourceUrl && (
        <span className="legal-source-chip">{data.sourceLabel}</span>
      )}

      <Handle type="source" position={Position.Right} className="legal-handle" />
    </div>
  );
}

const nodeTypes = {
  legalNode: LegalNode,
};

function isHiddenNode(node: ReasoningNode): boolean {
  const nodeType = node.node_type.toLowerCase();

  return nodeType === "question" || node.id.toLowerCase() === "question";
}

function findNodeByType(
  nodes: ReasoningNode[],
  acceptedTypes: string[],
): ReasoningNode | undefined {
  return nodes.find((node) =>
    acceptedTypes.includes(node.node_type.toLowerCase()),
  );
}

function findBranchTarget(
  reasoningMap: ReasoningMap,
  visibleNodeIds: Set<string>,
  label: "yes" | "no",
): ReasoningNode | undefined {
  const edge = reasoningMap.edges.find(
    (candidate) =>
      candidate.label?.toLowerCase() === label &&
      visibleNodeIds.has(candidate.target),
  );

  if (!edge) {
    return undefined;
  }

  return reasoningMap.nodes.find((node) => node.id === edge.target);
}

function toFlowNode(
  node: ReasoningNode,
  position: { x: number; y: number },
  forcedStepLabel?: string,
): LegalFlowNode {
  return {
    id: node.id,
    type: "legalNode",
    position,
    data: {
      kind: getNodeKind(node),
      stepLabel: forcedStepLabel ?? getDisplayStepLabel(node),
      title: getNodeTitle(node),
      sourceLabel: getSourceLabel(node),
      sourceUrl: node.source_url ?? undefined,
    },
    draggable: false,
  };
}

function buildDynamicFlowLayout(reasoningMap: ReasoningMap): {
  nodes: LegalFlowNode[];
  edges: Edge[];
} {
  const visibleReasoningNodes = reasoningMap.nodes.filter(
    (node) => !isHiddenNode(node),
  );

  const visibleNodeIds = new Set(visibleReasoningNodes.map((node) => node.id));

  const issueNode =
    findNodeByType(visibleReasoningNodes, ["issue"]) ?? visibleReasoningNodes[0];

  const conditionNode = findNodeByType(visibleReasoningNodes, ["condition"]);

  const testNode = findNodeByType(visibleReasoningNodes, [
    "test",
    "legal_test",
  ]);

  const yesNode =
    findBranchTarget(reasoningMap, visibleNodeIds, "yes") ??
    findNodeByType(visibleReasoningNodes, ["outcome"]);

  const noNode =
    findBranchTarget(reasoningMap, visibleNodeIds, "no") ??
    findNodeByType(visibleReasoningNodes, ["consequence", "remedy"]);

  const selectedNodeIds = new Set(
    [issueNode?.id, conditionNode?.id, testNode?.id, yesNode?.id, noNode?.id]
      .filter(Boolean)
      .map(String),
  );

  const extraNodes = visibleReasoningNodes.filter(
    (node) => !selectedNodeIds.has(node.id),
  );

  const flowNodes: LegalFlowNode[] = [];

  const hasDecisionLayout = Boolean(issueNode && conditionNode && testNode);

  if (hasDecisionLayout && issueNode && conditionNode && testNode) {
    flowNodes.push(toFlowNode(issueNode, { x: 0, y: 135 }));
    flowNodes.push(toFlowNode(conditionNode, { x: 285, y: 135 }));
    flowNodes.push(toFlowNode(testNode, { x: 570, y: 135 }));

    if (yesNode) {
      flowNodes.push(toFlowNode(yesNode, { x: 875, y: 45 }, "If yes"));
    }

    if (noNode) {
      flowNodes.push(toFlowNode(noNode, { x: 875, y: 225 }, "If no"));
    }

    extraNodes.slice(0, 2).forEach((node, index) => {
      flowNodes.push(
        toFlowNode(node, {
          x: 1175,
          y: 90 + index * 180,
        }),
      );
    });
  } else {
    visibleReasoningNodes.slice(0, 7).forEach((node, index) => {
      flowNodes.push(
        toFlowNode(node, {
          x: index * 285,
          y: 135,
        }),
      );
    });
  }

  const renderedNodeIds = new Set(flowNodes.map((node) => node.id));

  const flowEdges: Edge[] = reasoningMap.edges
    .filter(
      (edge) => renderedNodeIds.has(edge.source) && renderedNodeIds.has(edge.target),
    )
    .map((edge) => {
      const edgeLabel = edge.label?.trim();
      const normalizedLabel = edgeLabel?.toLowerCase();

      return {
        id: `${edge.source}-${edge.target}-${edgeLabel ?? "edge"}`,
        source: edge.source,
        target: edge.target,
        type: "smoothstep",
        label: edgeLabel || undefined,
        markerEnd: {
          type: MarkerType.ArrowClosed,
        },
        className:
          normalizedLabel === "yes"
            ? "legal-edge legal-edge--yes"
            : normalizedLabel === "no"
              ? "legal-edge legal-edge--no"
              : "legal-edge",
        labelBgPadding: [8, 4],
        labelBgBorderRadius: 999,
      };
    });

  if (flowEdges.length === 0 && flowNodes.length >= 2) {
    for (let index = 0; index < flowNodes.length - 1; index += 1) {
      flowEdges.push({
        id: `${flowNodes[index].id}-${flowNodes[index + 1].id}`,
        source: flowNodes[index].id,
        target: flowNodes[index + 1].id,
        type: "smoothstep",
        markerEnd: {
          type: MarkerType.ArrowClosed,
        },
        className: "legal-edge",
      });
    }
  }

  return {
    nodes: flowNodes,
    edges: flowEdges,
  };
}


export function LegalReasoningGraph({
  reasoningMap,
}: {
  reasoningMap?: ReasoningMap | null;
}) {
  const { nodes, edges } = useMemo(() => {
    if (!reasoningMap) {
      return {
        nodes: [],
        edges: [],
      };
    }

    return buildDynamicFlowLayout(reasoningMap);
  }, [reasoningMap]);

  if (!reasoningMap || nodes.length === 0) {
    return null;
  }

  return (
    <section className="legal-graph-section">
      <div className="legal-graph-header">
        <span className="legal-graph-badge">Legal reasoning map</span>
      </div>

      <div className="legal-graph-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.16 }}
          minZoom={0.35}
          maxZoom={1.3}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          panOnDrag
          zoomOnScroll={false}
          zoomOnPinch
          zoomOnDoubleClick={false}
          preventScrolling={false}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={20} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </section>
  );
}