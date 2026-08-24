import { describe, it, expect } from 'vitest';

import { createReconnectGuard } from '../src/hooks/reconnectGuard.ts';

describe('createReconnectGuard', () => {
  it('allows a current socket to reconnect after an unexpected close', () => {
    const guard = createReconnectGuard();
    const generation = guard.start();

    expect(guard.canReconnect(generation)).toBe(true);
  });

  it('blocks a late close callback after the socket lifecycle is stopped', () => {
    const guard = createReconnectGuard();
    const generation = guard.start();

    guard.stop();

    expect(guard.canReconnect(generation)).toBe(false);
  });

  it('invalidates an older socket when a new lifecycle starts', () => {
    const guard = createReconnectGuard();
    const oldGeneration = guard.start();
    const newGeneration = guard.start();

    expect(guard.canReconnect(oldGeneration)).toBe(false);
    expect(guard.canReconnect(newGeneration)).toBe(true);
  });
});