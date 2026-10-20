<script lang="ts">
	import ImageUpIcon from '@lucide/svelte/icons/image-up';
	import { cn } from '#lib/utils.js';
	import { ACCEPTED_TYPES, MAX_UPLOAD_MB } from '#lib/detections.js';

	let {
		onfile,
		disabled = false,
		compact = false
	}: { onfile: (file: File) => void; disabled?: boolean; compact?: boolean } = $props();

	let input: HTMLInputElement;
	let dragging = $state(false);

	function take(files: FileList | null | undefined) {
		const file = files?.[0];
		if (file) onfile(file);
	}
</script>

<label
	class={cn(
		'flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-input text-center transition-colors hover:border-primary/60 hover:bg-muted/40',
		compact ? 'px-4 py-4' : 'px-6 py-14',
		dragging && 'border-primary bg-primary/5',
		disabled && 'pointer-events-none opacity-60'
	)}
	ondragover={(e) => {
		e.preventDefault();
		dragging = true;
	}}
	ondragleave={() => (dragging = false)}
	ondrop={(e) => {
		e.preventDefault();
		dragging = false;
		take(e.dataTransfer?.files);
	}}
>
	<ImageUpIcon class={cn('text-muted-foreground', compact ? 'size-5' : 'size-10')} />
	<span class="text-sm font-medium">
		{compact
			? 'Drop or choose another image'
			: 'Drop a parasite egg image here, or click to choose'}
	</span>
	{#if !compact}
		<span class="text-xs text-muted-foreground">JPEG or PNG, up to {MAX_UPLOAD_MB} MB</span>
	{/if}
	<input
		bind:this={input}
		type="file"
		class="sr-only"
		accept={ACCEPTED_TYPES.join(',')}
		{disabled}
		data-testid="file-input"
		onchange={() => {
			take(input.files);
			input.value = '';
		}}
	/>
</label>
