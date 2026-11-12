package com.example.usgandroidapp;

/**
 * Minimal visibility state machine. It deliberately performs no prediction, filtering,
 * confidence hysteresis, or semantic target movement.
 */
public final class AntiBlinkStateMachine {
    public enum State { SEARCH, LOCKED, COAST, LOST }

    public static final class Snapshot {
        public final State state;
        public final YoloDetector.Detection authoritativeDetection;
        public final boolean freshResult;
        public final double ageMs;

        Snapshot(State state, YoloDetector.Detection detection, boolean freshResult,
                 double ageMs) {
            this.state = state;
            this.authoritativeDetection = detection;
            this.freshResult = freshResult;
            this.ageMs = ageMs;
        }

        public boolean isVisible() {
            return (state == State.LOCKED || state == State.COAST)
                    && authoritativeDetection != null;
        }
    }

    private final long coastDurationNs;
    private State state = State.SEARCH;
    private YoloDetector.Detection lastFresh;
    private long lastFreshNs;
    private boolean hasCoastedAtLeastOnce;

    public AntiBlinkStateMachine(long coastDurationMs) {
        if (coastDurationMs <= 0L) throw new IllegalArgumentException("coastDurationMs > 0");
        coastDurationNs = coastDurationMs * 1_000_000L;
    }

    public synchronized Snapshot onFreshInference(YoloDetector.Detection detection,
                                                   long completionTimestampNs) {
        if (detection != null) {
            lastFresh = detection;
            lastFreshNs = completionTimestampNs;
            state = State.LOCKED;
            hasCoastedAtLeastOnce = false;
            return snapshot(true, completionTimestampNs);
        }

        if (state == State.SEARCH || state == State.LOST || lastFresh == null) {
            state = (lastFresh == null && state == State.SEARCH) ? State.SEARCH : State.LOST;
            return snapshot(false, completionTimestampNs);
        }

        // The first missed update always enters COAST. Timeout validation starts on the
        // following completed inference, so one miss can never blank the presentation.
        if (state == State.LOCKED) {
            state = State.COAST;
            hasCoastedAtLeastOnce = true;
            return snapshot(false, completionTimestampNs);
        }

        long ageNs = Math.max(0L, completionTimestampNs - lastFreshNs);
        if (hasCoastedAtLeastOnce && ageNs > coastDurationNs) {
            state = State.LOST;
        } else {
            state = State.COAST;
        }
        return snapshot(false, completionTimestampNs);
    }

    private Snapshot snapshot(boolean fresh, long nowNs) {
        double ageMs = lastFreshNs > 0L ? Math.max(0L, nowNs - lastFreshNs) / 1_000_000.0 : 0.0;
        YoloDetector.Detection visible = (state == State.LOCKED || state == State.COAST)
                ? lastFresh : null;
        return new Snapshot(state, visible, fresh, ageMs);
    }

    public synchronized void reset() {
        state = State.SEARCH;
        lastFresh = null;
        lastFreshNs = 0L;
        hasCoastedAtLeastOnce = false;
    }
}
