import { describe, expect, it } from 'vitest';
import type { Detection } from './api';
import { countByClass, filterDetections, formatConf, validateFile } from './detections';

const det = (class_id: number, confidence: number): Detection => ({
	class_id,
	label: ['MIF', 'OV'][class_id],
	confidence,
	x1: 0,
	y1: 0,
	x2: 10,
	y2: 10
});

describe('filterDetections', () => {
	const all = [det(0, 0.9), det(1, 0.3), det(0, 0.15)];

	it('filters by confidence threshold (inclusive)', () => {
		expect(filterDetections(all, 0.3, new Set())).toEqual([all[0], all[1]]);
	});

	it('hides toggled-off classes', () => {
		expect(filterDetections(all, 0, new Set([0]))).toEqual([all[1]]);
	});
});

describe('countByClass', () => {
	it('counts every class, including empty ones', () => {
		expect(countByClass([det(0, 0.5), det(0, 0.6)], 2)).toEqual([2, 0]);
	});
});

describe('validateFile', () => {
	it('accepts a small jpeg', () => {
		expect(validateFile(new File(['x'], 'a.jpg', { type: 'image/jpeg' }))).toBeNull();
	});

	it('rejects other types', () => {
		expect(validateFile(new File(['x'], 'a.gif', { type: 'image/gif' }))).toMatch(/JPEG and PNG/);
	});

	it('rejects oversized files', () => {
		const big = new File([new Uint8Array(2 * 1024 * 1024)], 'a.png', { type: 'image/png' });
		expect(validateFile(big, 1)).toMatch(/at most 1 MB/);
	});
});

describe('formatConf', () => {
	it('formats as a whole percentage', () => {
		expect(formatConf(0.6289)).toBe('63%');
	});
});
