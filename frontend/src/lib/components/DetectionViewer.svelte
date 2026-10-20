<script lang="ts">
	import type { Detection } from '#lib/api.js';
	import { classColor, formatConf } from '#lib/detections.js';

	let {
		src,
		width,
		height,
		detections,
		img = $bindable()
	}: {
		src: string;
		width: number;
		height: number;
		detections: Detection[];
		img?: HTMLImageElement;
	} = $props();

	let hovered = $state<Detection | null>(null);

	// Stroke width in image pixels, so boxes look the same thickness at any image resolution.
	const stroke = $derived(Math.max(width, height) / 400);

	// Tooltip placement in rendered pixels: above the box, flipped inside it when there is no
	// room above, and clamped horizontally so it never leaves the (overflow-hidden) container.
	let boxW = $state(0);
	let boxH = $state(0);
	let tipW = $state(0);
	let tipH = $state(0);
	const GAP = 4;
	const tip = $derived.by(() => {
		if (!hovered) return { left: 0, top: 0 };
		const sx = boxW / width;
		const sy = boxH / height;
		const above = hovered.y1 * sy - tipH - GAP;
		return {
			left: Math.max(GAP, Math.min(hovered.x1 * sx, boxW - tipW - GAP)),
			top: above >= GAP ? above : hovered.y1 * sy + GAP
		};
	});
</script>

<div
	class="relative overflow-hidden rounded-lg border bg-black/5"
	bind:clientWidth={boxW}
	bind:clientHeight={boxH}
>
	<img bind:this={img} {src} alt="Uploaded micrograph" class="block h-auto w-full" />
	<svg
		class="absolute inset-0 size-full"
		viewBox="0 0 {width} {height}"
		preserveAspectRatio="none"
		data-testid="overlay"
	>
		{#each detections as d, i (i)}
			<rect
				role="img"
				aria-label="{d.label} {formatConf(d.confidence)}"
				x={d.x1}
				y={d.y1}
				width={d.x2 - d.x1}
				height={d.y2 - d.y1}
				fill={hovered === d ? classColor(d.class_id) : 'transparent'}
				fill-opacity="0.15"
				stroke={classColor(d.class_id)}
				stroke-width={hovered === d ? stroke * 1.6 : stroke}
				class="cursor-pointer transition-[stroke-width]"
				pointer-events="all"
				onmouseenter={() => (hovered = d)}
				onmouseleave={() => (hovered = null)}
			/>
		{/each}
	</svg>

	{#if hovered}
		<div
			bind:offsetWidth={tipW}
			bind:offsetHeight={tipH}
			class="pointer-events-none absolute z-10 rounded-md border bg-popover px-2 py-1 text-xs font-medium whitespace-nowrap text-popover-foreground shadow-md"
			style:left="{tip.left}px"
			style:top="{tip.top}px"
			data-testid="tooltip"
		>
			<span
				class="mr-1 inline-block size-2 rounded-full"
				style:background={classColor(hovered.class_id)}
			></span>
			{hovered.label} · {formatConf(hovered.confidence)}
		</div>
	{/if}
</div>
