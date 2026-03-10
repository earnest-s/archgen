import React from "react";

interface Props {
  /** Base64-encoded PNG string returned by the backend (no data-URI prefix needed). */
  diagramBase64?: string;
  /** Alt text for the diagram image. */
  alt?: string;
}

/**
 * DiagramViewer
 *
 * Renders the PNG diagram returned by the backend. Accepts a raw base64
 * string and wraps it in a data-URI so the browser can display it inline
 * without a separate HTTP request.
 */
export const DiagramViewer: React.FC<Props> = ({
  diagramBase64,
  alt = "Architecture diagram",
}) => {
  if (!diagramBase64) {
    return (
      <div
        className="flex h-64 w-full items-center justify-center rounded-xl
                     border-2 border-dashed border-gray-300 text-gray-400 text-sm"
        aria-label="Diagram placeholder"
      >
        Diagram will appear here
      </div>
    );
  }

  const src = `data:image/png;base64,${diagramBase64}`;

  return (
    <div className="w-full overflow-auto rounded-xl border border-gray-200 bg-white shadow-sm p-2">
      <img
        src={src}
        alt={alt}
        className="mx-auto max-w-full object-contain"
        style={{ maxHeight: "70vh" }}
      />
    </div>
  );
};

export default DiagramViewer;
