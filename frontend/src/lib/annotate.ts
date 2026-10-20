import type { Detection } from './api';
import { classColor, formatConf } from './detections';

/** Draw `detections` onto a full-resolution copy of `img` and return it as a PNG blob. */
export async function renderAnnotatedPng(
	img: HTMLImageElement,
	detections: Detection[]
): Promise<Blob> {
	const canvas = document.createElement('canvas');
	canvas.width = img.naturalWidth;
	canvas.height = img.naturalHeight;
	const ctx = canvas.getContext('2d');
	if (!ctx) throw new Error('Canvas 2D context is not available');
	ctx.drawImage(img, 0, 0);

	// Scale strokes and text with the image so they stay legible on large micrographs.
	const unit = Math.max(canvas.width, canvas.height) / 640;
	ctx.lineWidth = 2 * unit;
	ctx.font = `600 ${12 * unit}px Inter, system-ui, sans-serif`;
	ctx.textBaseline = 'top';

	// Ascending confidence, so the most confident label is drawn on top where boxes overlap.
	for (const d of detections.toSorted((a, b) => a.confidence - b.confidence)) {
		const color = classColor(d.class_id);
		ctx.strokeStyle = color;
		ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);

		const text = `${d.label} ${formatConf(d.confidence)}`;
		const pad = 3 * unit;
		const h = 12 * unit + 2 * pad;
		const w = ctx.measureText(text).width + 2 * pad;
		// Keep the label inside the image: above the box if it fits, else inside; clamp horizontally.
		const x = Math.max(0, Math.min(d.x1, canvas.width - w));
		const y = d.y1 - h >= 0 ? d.y1 - h : d.y1;
		ctx.fillStyle = color;
		ctx.fillRect(x, y, w, h);
		ctx.fillStyle = '#fff';
		ctx.fillText(text, x + pad, y + pad);
	}

	return new Promise((resolve, reject) =>
		canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG export failed'))), 'image/png')
	);
}
