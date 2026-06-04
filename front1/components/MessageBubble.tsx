import { Message, RetrievalDocument, TextBlock, ToolCallBlock, WorkflowTraceBlock, IntentBlock, PlanBlock, StepBlock, RetrievalBlock } from '../types';
import { CollapsibleSection } from './CollapsibleSection';
import { DocumentFragmentModal } from './DocumentFragmentModal';
import { ToolCard } from './ToolCard';
import { WorkflowTraceCard } from './WorkflowTraceCard';
import { IntentCard } from './IntentCard';
import { PlanCard } from './PlanCard';
import { StepCard } from './StepCard';
import { RetrievalCard } from './RetrievalCard';
import { renderMarkdown } from '../utils/markdown';
import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';

interface MessageBubbleProps {
  message: Message;
  onToggleToolExpand?: (messageId: string, blockId: string) => void;
}

export function MessageBubble({ message, onToggleToolExpand }: MessageBubbleProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [selectedCitationDocument, setSelectedCitationDocument] = useState<RetrievalDocument | null>(null);

  const citationDocumentMap = useMemo(() => {
    const mapping = new Map<number, RetrievalDocument>();
    for (const block of message.blocks || []) {
      if (block.type !== 'retrieval') {
        continue;
      }
      for (const document of block.documents || []) {
        if (typeof document.citation_id === 'number') {
          mapping.set(document.citation_id, document);
        }
      }
    }
    return mapping;
  }, [message.blocks]);

  const handleCitationClick = (event: MouseEvent<HTMLDivElement>) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const citationElement = target.closest('[data-citation-id]');
    if (!(citationElement instanceof HTMLElement)) {
      return;
    }
    const citationId = Number(citationElement.dataset.citationId || '');
    if (!Number.isFinite(citationId)) {
      return;
    }
    const document = citationDocumentMap.get(citationId);
    if (document) {
      event.preventDefault();
      setSelectedCitationDocument(document);
    }
  };

  useEffect(() => {
    if (!contentRef.current) {
      return;
    }
    if (message.chunks) {
      const lastChunk = message.chunks[message.chunks.length - 1];
      if (lastChunk?.isStreaming && lastChunk.type === 'content') {
        contentRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
    if (message.blocks && message.isStreaming) {
      contentRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [message.chunks, message.blocks, message.isStreaming]);

  if (message.role === 'user') {
    return (
      <div className="flex items-end mb-6 fade-in justify-end">
        <div className="bg-indigo-600 text-white p-3 rounded-2xl rounded-tr-none shadow-md max-w-[80%] text-sm">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.blocks) {
    const blocks = message.blocks;
    return (
      <div className="flex items-start fade-in mb-8">
        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex-shrink-0 flex items-center justify-center text-white text-xs mr-3 mt-1 shadow-md">
          AI
        </div>
        <div className="flex-1 min-w-0">
          {blocks.map((block) => {
            if (block.type === 'text') {
              const textBlock = block as TextBlock;
              return (
                <div
                  key={textBlock.id}
                  ref={contentRef}
                  className="prose prose-sm prose-slate max-w-none text-slate-700 bg-white p-5 rounded-xl rounded-tl-none shadow-sm border border-slate-100 mb-3"
                >
                  <div
                    onClick={handleCitationClick}
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(textBlock.content) }}
                  />
                  {message.isStreaming && block === blocks[blocks.length - 1] && (
                    <span className="inline-block w-2 h-4 bg-indigo-600 ml-1 animate-pulse"></span>
                  )}
                </div>
              );
            }

            if (block.type === 'workflow') {
              const workflowBlock = block as WorkflowTraceBlock;
              return (
                <WorkflowTraceCard
                  key={workflowBlock.id}
                  stage={workflowBlock.stage}
                  message={workflowBlock.message}
                  status={workflowBlock.status}
                  route={workflowBlock.route}
                />
              );
            }

            if (block.type === 'tool_call') {
              const toolBlock = block as ToolCallBlock;
              return (
                <ToolCard
                  key={toolBlock.id}
                  toolName={toolBlock.toolName}
                  input={toolBlock.input}
                  output={toolBlock.output}
                  status={toolBlock.status}
                  isExpanded={toolBlock.isExpanded}
                  onToggleExpand={
                    onToggleToolExpand
                      ? () => onToggleToolExpand(message.id, toolBlock.id)
                      : undefined
                  }
                />
              );
            }

            if (block.type === 'intent') {
              const intentBlock = block as IntentBlock;
              return (
                <IntentCard
                  key={intentBlock.id}
                  mode={intentBlock.mode}
                  reason={intentBlock.reason}
                  candidateTools={intentBlock.candidateTools}
                />
              );
            }

            if (block.type === 'plan') {
              const planBlock = block as PlanBlock;
              return (
                <PlanCard
                  key={planBlock.id}
                  rationale={planBlock.rationale}
                  steps={planBlock.steps}
                />
              );
            }

            if (block.type === 'step') {
              const stepBlock = block as StepBlock;
              return (
                <StepCard
                  key={stepBlock.id}
                  stepId={stepBlock.stepId}
                  tool={stepBlock.tool}
                  goal={stepBlock.goal}
                  args={stepBlock.args}
                  output={stepBlock.output}
                  error={stepBlock.error}
                  durationMs={stepBlock.durationMs}
                  status={stepBlock.status}
                />
              );
            }

            if (block.type === 'retrieval') {
              const retrievalBlock = block as RetrievalBlock;
              return (
                <RetrievalCard
                  key={retrievalBlock.id}
                  query={retrievalBlock.query}
                  strategy={retrievalBlock.strategy}
                  documents={retrievalBlock.documents}
                />
              );
            }

            return null;
          })}

          <DocumentFragmentModal
            document={selectedCitationDocument}
            isOpen={selectedCitationDocument != null}
            onClose={() => setSelectedCitationDocument(null)}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start fade-in mb-8">
      <div className="w-8 h-8 rounded-lg bg-indigo-600 flex-shrink-0 flex items-center justify-center text-white text-xs mr-3 mt-1 shadow-md">
        AI
      </div>
      <div className="flex-1 min-w-0">
        {message.chunks && message.chunks.map((chunk) => {
          if (chunk.type === 'thought') {
            return (
              <CollapsibleSection
                key={chunk.id}
                type="thought"
                content={chunk.content}
                defaultCollapsed={true}
              />
            );
          }

          if (chunk.type === 'tool_call') {
            return (
              <CollapsibleSection
                key={chunk.id}
                type="tool_call"
                content={chunk.content}
                defaultCollapsed={true}
              />
            );
          }

          if (chunk.type === 'content') {
            return (
              <div
                key={chunk.id}
                ref={contentRef}
                className="prose prose-sm prose-slate max-w-none text-slate-700 bg-white p-5 rounded-xl rounded-tl-none shadow-sm border border-slate-100 mb-3"
              >
                <div
                  onClick={handleCitationClick}
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(chunk.content) }}
                />
                {chunk.isStreaming && (
                  <span className="inline-block w-2 h-4 bg-indigo-600 ml-1 animate-pulse"></span>
                )}
              </div>
            );
          }

          return null;
        })}

        <DocumentFragmentModal
          document={selectedCitationDocument}
          isOpen={selectedCitationDocument != null}
          onClose={() => setSelectedCitationDocument(null)}
        />
      </div>
    </div>
  );
}