<script setup lang="ts">
import { isDeepEqual, isUserInDarkMode } from "@/services/utils";
import { View as VegaView } from "vega";
import embed, { type Config, type VisualizationSpec } from "vega-embed";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";

const props = defineProps<{ spec: VisualizationSpec }>();

const container = ref<HTMLDivElement>();
const font = "GeistMono, monospace";
const vegaConfig: Config = {
  axis: { labelFont: font, titleFont: font },
  legend: { labelFont: font, titleFont: font },
  header: { labelFont: font, titleFont: font },
  mark: { font: font },
  title: { font: font, subtitleFont: font },
  background: "transparent",
};
let vegaView: VegaView | null = null;

async function makePlot(spec: VisualizationSpec) {
  const mySpec = { ...spec, width: "container" } as VisualizationSpec;
  const r = await embed(container.value!, mySpec, {
    theme: isUserInDarkMode() ? "dark" : undefined,
    config: vegaConfig,
    actions: false,
  });
  vegaView = r.view;
}
onMounted(async () => {
  if (container.value) {
    makePlot(props.spec);
  }
});

onBeforeUnmount(() => {
  if (vegaView) {
    vegaView.finalize();
  }
});

watch(
  () => props.spec,
  async (newSpec, oldSpec) => {
    if (!isDeepEqual(newSpec, oldSpec)) {
      makePlot(newSpec);
    }
  }
);
</script>

<template>
  <div ref="container"></div>
</template>

<style scoped>
div {
  width: 100%;
}
</style>
