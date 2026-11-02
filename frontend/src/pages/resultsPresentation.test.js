import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildMetricPresentations,
  buildPersonalizedFeedback,
  getRecommendationPresentation,
} from './resultsPresentation.js';

test('all recommendations have clear consumer-facing copy and stable status classes', () => {
  const expected = {
    KEEP: ['Excellent photo', 'Your photo looks good.', 'keep'],
    RECOVER: ['Good photo, with room to improve', 'Some areas could be improved.', 'recover'],
    RETAKE: ['Consider taking this photo again', 'A new shot may give you a better result.', 'retake'],
  };
  for (const [recommendation, copy] of Object.entries(expected)) {
    assert.deepEqual(Object.values(getRecommendationPresentation(recommendation)), copy);
  }
});

test('metric summaries are category-aware across the four photography categories', () => {
  const categories = ['Portrait Photography', 'Item Photography', 'Landscape Photography', 'General Photography'];
  const features = {
    blur: { value: 8, focus_discrepancy: false },
    brightness: { value: 52 },
    contrast: { value: 50 },
    noise: { value: 8 },
    face_visibility: {
      value: 85,
      face_count: 1,
      applicable_to_quality: true,
    },
    object_visibility: {
      value: 85,
      object_detected: true,
      applicable_to_quality: true,
    },
    composition: { value: 80, subject_type: 'Visual Saliency Centroid' },
  };
  const memberships = {
    blur: { Low: 1, Medium: 0, High: 0 },
    brightness: { Dark: 0, Normal: 1, Overexposed: 0 },
    contrast: { Low: 0, Medium: 1, High: 0 },
    noise: { Low: 1, Medium: 0, High: 0 },
    face_visibility: { Poor: 0, Acceptable: 0, Clear: 1 },
    object_visibility: { Poor: 0, Acceptable: 0, Clear: 1 },
    composition: { Poor: 0, Average: 0, Good: 1 },
  };

  for (const category of categories) {
    const metrics = buildMetricPresentations(features, memberships, category);
    assert.equal(
      metrics.length,
      ['Portrait Photography', 'Item Photography'].includes(category) ? 6 : 5,
    );
    const face = metrics.find((metric) => metric.key === 'face_visibility');
    assert.equal(Boolean(face), category === 'Portrait Photography');
    if (face) assert.match(face.explanation, /1 face detected/);
    const object = metrics.find((metric) => metric.key === 'object_visibility');
    assert.equal(Boolean(object), category === 'Item Photography');
    assert.ok(metrics.every((metric) => metric.status && metric.explanation));
  }
});

test('all recommendation states retain their presentation across every category', () => {
  const categories = ['Portrait Photography', 'Item Photography', 'Landscape Photography', 'General Photography'];
  const features = {
    blur: { value: 8, focus_discrepancy: false },
    brightness: { value: 52 },
    contrast: { value: 50 },
    noise: { value: 8 },
    face_visibility: { value: 85, face_count: 1, applicable_to_quality: true },
    object_visibility: { value: 85, object_detected: true, applicable_to_quality: true },
    composition: { value: 80, subject_type: 'Visual Saliency Centroid' },
  };
  const memberships = {
    blur: { Low: 1, Medium: 0, High: 0 },
    brightness: { Dark: 0, Normal: 1, Overexposed: 0 },
    contrast: { Low: 0, Medium: 1, High: 0 },
    noise: { Low: 1, Medium: 0, High: 0 },
    face_visibility: { Poor: 0, Acceptable: 0, Clear: 1 },
    object_visibility: { Poor: 0, Acceptable: 0, Clear: 1 },
    composition: { Poor: 0, Average: 0, Good: 1 },
  };
  for (const category of categories) {
    for (const recommendation of ['KEEP', 'RECOVER', 'RETAKE']) {
      const presentation = getRecommendationPresentation(recommendation);
      assert.ok(presentation.title);
      assert.ok(presentation.description);
      assert.ok(['keep', 'recover', 'retake'].includes(presentation.className));
      const metrics = buildMetricPresentations(features, memberships, category);
      assert.equal(
        metrics.length,
        ['Portrait Photography', 'Item Photography'].includes(category) ? 6 : 5,
      );
      assert.equal(Boolean(metrics.find((metric) => metric.key === 'face_visibility')),
        category === 'Portrait Photography');
      assert.equal(Boolean(metrics.find((metric) => metric.key === 'object_visibility')),
        category === 'Item Photography');
      assert.ok(Array.isArray(buildPersonalizedFeedback(metrics, recommendation)));
    }
  }
});

test('personalized feedback reflects blur, underexposure, and grain measurements', () => {
  const metrics = buildMetricPresentations(
    {
      blur: { value: 92 },
      brightness: { value: 20 },
      contrast: { value: 50 },
      noise: { value: 82 },
      face_visibility: { value: 0, face_count: 0, applicable_to_quality: false },
      composition: { value: 60, subject_type: 'Visual Saliency Centroid' },
    },
    {
      blur: { Low: 0, Medium: 0, High: 1 },
      brightness: { Dark: 1, Normal: 0, Overexposed: 0 },
      contrast: { Low: 0, Medium: 1, High: 0 },
      noise: { Low: 0, Medium: 0, High: 1 },
      composition: { Poor: 0, Average: 1, Good: 0 },
    },
    'Landscape Photography',
  );
  const feedback = buildPersonalizedFeedback(metrics);
  assert.deepEqual(feedback.map((item) => item.title), [
    'Blur detected',
    'Lighting is low',
    'Visible image grain',
  ]);
  assert.ok(feedback.every((item) => item.tone === 'attention'));
});

test('saved results without fuzzy memberships use their existing CV descriptions', () => {
  const metrics = buildMetricPresentations({
    blur: { value: 95, sharpness: 5, description: 'Low sharpness / Blurry' },
    brightness: { value: 24, description: 'Dark / Underexposed' },
    contrast: { value: 80, description: 'High Contrast' },
    noise: { value: 90, description: 'High Noise' },
    composition: { value: 30, description: 'Poor (Awkward framing)' },
    face_visibility: {
      value: 0,
      face_count: 0,
      applicable_to_quality: false,
      description: 'Not assessed',
    },
    object_visibility: {
      value: 50,
      object_detected: false,
      applicable_to_quality: false,
      description: 'Object visibility not assessed',
    },
  }, {}, 'Item Photography');
  const byKey = Object.fromEntries(metrics.map((metric) => [metric.key, metric]));
  assert.equal(byKey.blur.status, 'Poor');
  assert.match(byKey.blur.explanation, /Significant blur/);
  assert.equal(byKey.brightness.status, 'Poor');
  assert.match(byKey.brightness.explanation, /underexposed/i);
  assert.equal(byKey.contrast.status, 'Fair');
  assert.match(byKey.contrast.explanation, /strongly separated/);
  assert.equal(byKey.noise.status, 'Poor');
  assert.equal(byKey.composition.status, 'Poor');
  assert.equal(byKey.object_visibility.status, 'Not assessed');
  assert.match(byKey.object_visibility.explanation, /left neutral/i);
});

test('strong image results receive positive feedback rather than fabricated issues', () => {
  const metrics = buildMetricPresentations(
    {
      blur: { value: 8 },
      brightness: { value: 52 },
      contrast: { value: 50 },
      noise: { value: 8 },
      face_visibility: { value: 0, face_count: 0, applicable_to_quality: false },
      composition: { value: 80, subject_type: 'Visual Saliency Centroid' },
    },
    {
      blur: { Low: 1, Medium: 0, High: 0 },
      brightness: { Dark: 0, Normal: 1, Overexposed: 0 },
      contrast: { Low: 0, Medium: 1, High: 0 },
      noise: { Low: 1, Medium: 0, High: 0 },
      composition: { Poor: 0, Average: 0, Good: 1 },
    },
    'Landscape Photography',
  );
  const feedback = buildPersonalizedFeedback(metrics);
  assert.ok(feedback.length > 0);
  assert.ok(feedback.every((item) => item.tone === 'positive'));
  assert.ok(feedback.every((item) => !/face|noise|blur detected|underexposed/i.test(item.title)));
});
