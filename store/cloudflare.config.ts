// Deployed with the `cf` CLI: `cf deploy` from this directory (see README.md).
// A plain object, so that nothing has to be installed to read it. The D1 id
// below is this project's database: to deploy your own store, replace it with
// the id `cf d1 create` printed for yours. Tokens are
// secrets named TOKEN_<SOURCE>, one per client, uploaded with --secrets-file.
export default {
  worker: {
    name: "gpubox-reputation",
    entrypoint: "./src/worker.js",
    compatibilityDate: "2025-09-01", // the tests run the Worker under the same date
    env: {
      DB: { type: "d1", name: "gpubox-reputation", id: "e95b58ad-c652-47df-a210-b9c68f4c3a6b" },
    },
    triggers: [{ type: "scheduled", schedule: "17 4 * * *" }],
  },
};
