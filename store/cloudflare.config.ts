// Deployed with the `cf` CLI: `cf deploy` from this directory (see README.md).
// A plain object, so that nothing has to be installed to read it. Tokens are
// secrets named TOKEN_<SOURCE>, one per client, uploaded with --secrets-file.
export default {
  worker: {
    name: "gpubox-reputation",
    entrypoint: "./src/worker.js",
    compatibilityDate: "2026-09-17",
    env: {
      DB: { type: "d1", name: "gpubox-reputation", id: "<database-uuid>" },
    },
    triggers: [{ type: "scheduled", schedule: "17 4 * * *" }],
  },
};
