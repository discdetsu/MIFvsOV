<script lang="ts">
	import { onMount } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import DownloadIcon from '@lucide/svelte/icons/download';
	import LoaderIcon from '@lucide/svelte/icons/loader-circle';
	import CircleAlertIcon from '@lucide/svelte/icons/circle-alert';
	import { detect, getModelInfo, type DetectResponse, type ModelInfo } from '#lib/api.js';
	import { renderAnnotatedPng } from '#lib/annotate.js';
	import {
		classColor,
		countByClass,
		filterDetections,
		formatConf,
		validateFile
	} from '#lib/detections.js';
	import Dropzone from '#lib/components/Dropzone.svelte';
	import DetectionViewer from '#lib/components/DetectionViewer.svelte';
	import * as Card from '#lib/components/ui/card/index.js';
	import * as Alert from '#lib/components/ui/alert/index.js';
	import { Button } from '#lib/components/ui/button/index.js';
	import { Slider } from '#lib/components/ui/slider/index.js';
	import { Toggle } from '#lib/components/ui/toggle/index.js';

	let model = $state<ModelInfo | null>(null);
	let fileName = $state('');
	let previewUrl = $state<string | null>(null);
	let result = $state<DetectResponse | null>(null);
	let loading = $state(false);
	let error = $state<string | null>(null);
	let minConf = $state(0.25);
	const hidden = new SvelteSet<number>();
	let img = $state<HTMLImageElement>();
	let controller: AbortController | null = null;

	const names = $derived(result?.model.names ?? model?.names ?? ['MIF', 'OV']);
	const visible = $derived(result ? filterDetections(result.detections, minConf, hidden) : []);
	const counts = $derived(countByClass(visible, names.length));

	onMount(() => {
		getModelInfo()
			.then((info) => {
				model = info;
				minConf = info.conf;
			})
			.catch(() => {
				// The page still works with defaults; a real problem will surface on upload.
			});
		return () => previewUrl && URL.revokeObjectURL(previewUrl);
	});

	async function onfile(file: File) {
		const invalid = validateFile(file);
		if (invalid) {
			error = invalid;
			return;
		}
		controller?.abort();
		controller = new AbortController();
		if (previewUrl) URL.revokeObjectURL(previewUrl);
		previewUrl = URL.createObjectURL(file);
		fileName = file.name;
		result = null;
		error = null;
		loading = true;
		try {
			result = await detect(file, controller.signal);
		} catch (e) {
			if ((e as Error).name !== 'AbortError') error = (e as Error).message;
		} finally {
			loading = false;
		}
	}

	async function download() {
		if (!img || !result) return;
		const blob = await renderAnnotatedPng(img, visible);
		const a = document.createElement('a');
		a.href = URL.createObjectURL(blob);
		a.download = `${fileName.replace(/\.[^.]+$/, '') || 'image'}-annotated.png`;
		a.click();
		URL.revokeObjectURL(a.href);
	}

	function toggleClass(id: number, pressed: boolean) {
		if (pressed) hidden.delete(id);
		else hidden.add(id);
	}
</script>

<svelte:head>
	<title>MIF & OV Detection</title>
</svelte:head>

<div class="mx-auto flex min-h-dvh max-w-6xl flex-col px-4 py-8 sm:px-6">
	<header class="mb-8 flex items-center gap-4">
		<img src="/ov-logo.png" alt="" class="size-14 rounded-full object-cover shadow-sm" />
		<div>
			<h1 class="text-xl font-semibold tracking-tight sm:text-2xl">MIF & OV Egg Detection</h1>
			<p class="text-sm text-muted-foreground">
				Minute intestinal fluke vs <i>Opisthorchis viverrini</i> egg detection in microscopic images
			</p>
		</div>
	</header>

	<main class="flex-1">
		{#if error}
			<Alert.Root variant="destructive" class="mb-4">
				<CircleAlertIcon />
				<Alert.Title>Detection failed</Alert.Title>
				<Alert.Description>{error}</Alert.Description>
			</Alert.Root>
		{/if}

		{#if !previewUrl}
			<Card.Root>
				<Card.Content>
					<Dropzone {onfile} />
				</Card.Content>
			</Card.Root>
		{:else}
			<div class="grid gap-6 lg:grid-cols-[1fr_300px]">
				<div class="min-w-0">
					{#if result}
						<DetectionViewer
							src={previewUrl}
							width={result.image.width}
							height={result.image.height}
							detections={visible}
							bind:img
						/>
					{:else}
						<div class="relative overflow-hidden rounded-lg border">
							<img
								src={previewUrl}
								alt="Uploaded micrograph"
								class="block h-auto w-full opacity-50"
							/>
							{#if loading}
								<div class="absolute inset-0 grid place-items-center">
									<span
										class="flex items-center gap-2 rounded-full bg-background/90 px-4 py-2 text-sm shadow"
									>
										<LoaderIcon class="size-4 animate-spin" /> Detecting eggs…
									</span>
								</div>
							{/if}
						</div>
					{/if}
					<p class="mt-2 truncate text-xs text-muted-foreground">
						{[
							fileName,
							result && `${result.image.width}×${result.image.height}`,
							result && `${result.inference_ms} ms`
						]
							.filter(Boolean)
							.join(' · ')}
					</p>
				</div>

				<aside class="flex flex-col gap-4">
					<Card.Root>
						<Card.Header>
							<Card.Title>Detections</Card.Title>
							<Card.Description>
								{#if result}
									{visible.length} of {result.detections.length} shown
								{:else}
									Waiting for results…
								{/if}
							</Card.Description>
						</Card.Header>
						<Card.Content class="flex flex-col gap-5">
							<div class="flex flex-col gap-3">
								<div class="flex items-center justify-between text-sm">
									<span class="font-medium" id="conf-label">Min. confidence</span>
									<span class="text-muted-foreground tabular-nums" data-testid="conf-value"
										>{formatConf(minConf)}</span
									>
								</div>
								<Slider
									type="single"
									bind:value={minConf}
									min={model?.detect_conf ?? 0.1}
									max={1}
									step={0.01}
									aria-labelledby="conf-label"
									disabled={!result}
								/>
							</div>

							<div class="flex flex-col gap-2">
								<span class="text-sm font-medium">Classes</span>
								{#each names as name, id (id)}
									<Toggle
										variant="outline"
										class="justify-between"
										pressed={!hidden.has(id)}
										onPressedChange={(p) => toggleClass(id, p)}
										disabled={!result}
										aria-label="Show {name}"
									>
										<span class="flex items-center gap-2">
											<span
												class="size-3 rounded-sm"
												style:background={hidden.has(id) ? 'transparent' : classColor(id)}
												style:outline="2px solid {classColor(id)}"
											></span>
											{name}
										</span>
										<span class="tabular-nums" data-testid="count-{name}">{counts[id]}</span>
									</Toggle>
								{/each}
							</div>

							{#if result && result.detections.length === 0}
								<p class="text-sm text-muted-foreground">No eggs detected in this image.</p>
							{/if}
						</Card.Content>
						<Card.Footer>
							<Button class="w-full" onclick={download} disabled={!result}>
								<DownloadIcon /> Download annotated PNG
							</Button>
						</Card.Footer>
					</Card.Root>

					<Dropzone {onfile} disabled={loading} compact />
				</aside>
			</div>
		{/if}
	</main>

	<footer class="mt-12 border-t pt-6 text-center text-xs leading-relaxed text-muted-foreground">
		<p class="font-medium">Kasetsart University</p>
		<p>
			YOLOv5 model trained on data from Phramongkutklao Hospital's laboratory and the
			<a
				class="underline underline-offset-2"
				href="https://dx.doi.org/10.21227/vyh8-4h71"
				target="_blank"
				rel="noreferrer">ICIP 2022 challenge dataset</a
			>. For research use only, not for diagnosis.
		</p>
	</footer>
</div>
