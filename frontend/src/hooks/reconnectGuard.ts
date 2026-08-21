export interface ReconnectGuard {
  start(): number;
  isActive(): boolean;
  canReconnect(generation: number): boolean;
  stop(): void;
}

/**
 * Prevents a socket that is already being torn down from scheduling a new
 * connection after its owning component has unmounted or been replaced.
 */
export function createReconnectGuard(): ReconnectGuard {
  let active = false;
  let generation = 0;

  return {
    start() {
      active = true;
      generation += 1;
      return generation;
    },

    isActive() {
      return active;
    },

    canReconnect(candidate) {
      return active && candidate === generation;
    },

    stop() {
      active = false;
      generation += 1;
    },
  };
}
