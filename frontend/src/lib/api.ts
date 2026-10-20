import type { components } from './api-types';

export type Detection = components['schemas']['Detection'];
export type DetectResponse = components['schemas']['DetectResponse'];
export type ModelInfo = components['schemas']['ModelInfo'];

export class ApiError extends Error {
	constructor(
		public status: number,
		message: string
	) {
		super(message);
	}
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(path, init);
	if (!res.ok) {
		let message = `Request failed (${res.status})`;
		try {
			const body = await res.json();
			if (typeof body.detail === 'string') message = body.detail;
		} catch {
			// non-JSON error body
		}
		throw new ApiError(res.status, message);
	}
	return res.json() as Promise<T>;
}

export function getModelInfo(): Promise<ModelInfo> {
	return request('/api/model');
}

export function detect(file: File, signal?: AbortSignal): Promise<DetectResponse> {
	const body = new FormData();
	body.append('file', file);
	return request('/api/detect', { method: 'POST', body, signal });
}
