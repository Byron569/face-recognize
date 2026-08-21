import test from 'node:test';
import assert from 'node:assert/strict';

import { createReconnectGuard } from '../src/hooks/reconnectGuard.ts';

test('allows a current socket to reconnect after an unexpected close', () => {
  const guard = createReconnectGuard();
  const generation = guard.start();

  assert.equal(guard.canReconnect(generation), true);
});

test('blocks a late close callback after the socket lifecycle is stopped', () => {
  const guard = createReconnectGuard();
  const generation = guard.start();

  guard.stop();

  assert.equal(guard.canReconnect(generation), false);
});

test('invalidates an older socket when a new lifecycle starts', () => {
  const guard = createReconnectGuard();
  const oldGeneration = guard.start();
  const newGeneration = guard.start();

  assert.equal(guard.canReconnect(oldGeneration), false);
  assert.equal(guard.canReconnect(newGeneration), true);
});
