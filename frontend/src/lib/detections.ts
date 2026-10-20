import type { Detection } from './api';

export const ACCEPTED_TYPES = ['image/jpeg', 'image/png'];
export const MAX_UPLOAD_MB = 5;

/** Per-class box colour, indexed by class_id (0 = MIF, 1 = OV). */
export const CLASS_COLORS = ['#e8590c', '#1c7ed6'];

export function classColor(classId: number): string {
	return CLASS_COLORS[classId % CLASS_COLORS.length];
}

/** Returns an error message, or null if the file can be uploaded. */
export function validateFile(file: File, maxMb = MAX_UPLOAD_MB): string | null {
	if (!ACCEPTED_TYPES.includes(file.type)) return 'Only JPEG and PNG images are supported.';
	if (file.size > maxMb * 1024 * 1024) return `Image must be at most ${maxMb} MB.`;
	return null;
}

export function filterDetections(
	detections: Detection[],
	minConf: number,
	hidden: ReadonlySet<number>
): Detection[] {
	return detections.filter((d) => d.confidence >= minConf && !hidden.has(d.class_id));
}

export function countByClass(detections: Detection[], numClasses: number): number[] {
	const counts = Array<number>(numClasses).fill(0);
	for (const d of detections) counts[d.class_id]++;
	return counts;
}

export function formatConf(conf: number): string {
	return `${Math.round(conf * 100)}%`;
}
