const recommendationCopy = {
  KEEP: {
    title: 'Excellent photo',
    description: 'Your photo looks good.',
    className: 'keep',
  },
  RECOVER: {
    title: 'Good photo, with room to improve',
    description: 'Some areas could be improved.',
    className: 'recover',
  },
  RETAKE: {
    title: 'Consider taking this photo again',
    description: 'A new shot may give you a better result.',
    className: 'retake',
  },
};

const metricDefinitions = {
  blur: { label: 'Focus and Clarity', terms: { Low: 'Good', Medium: 'Fair', High: 'Poor' } },
  brightness: { label: 'Lighting', terms: { Dark: 'Poor', Normal: 'Good', Overexposed: 'Poor' } },
  contrast: { label: 'Image Contrast', terms: { Low: 'Poor', Medium: 'Good', High: 'Fair' } },
  noise: { label: 'Image Grain', terms: { Low: 'Good', Medium: 'Fair', High: 'Poor' } },
  face_visibility: { label: 'Face Detection / Face Visibility', terms: { Poor: 'Poor', Acceptable: 'Fair', Clear: 'Good' } },
  object_visibility: { label: 'Primary Object Visibility', terms: { Poor: 'Poor', Acceptable: 'Fair', Clear: 'Good' } },
  composition: { label: 'Framing', terms: { Poor: 'Poor', Average: 'Fair', Good: 'Good' } },
};

const statusClass = (status) => status.toLowerCase().replaceAll(' ', '-');

function strongestTerm(degrees) {
  if (!degrees || typeof degrees !== 'object') return null;
  return Object.entries(degrees)
    .filter(([, degree]) => Number.isFinite(degree))
    .sort((left, right) => right[1] - left[1])[0]?.[0] ?? null;
}

function fallbackStatus(key, item, category) {
  const description = String(item?.description ?? '').toLowerCase();
  if (key === 'blur') {
    if (description.includes('low sharpness') || description.includes('blurry')) return 'Poor';
    if (description.includes('moderate sharpness')) return 'Fair';
    if (description.includes('high sharpness')) return 'Good';
  }
  if (key === 'brightness') {
    if (description.includes('underexposed') || description.includes('overexposed') || description.includes('blown')) return 'Poor';
    if (description.includes('balanced exposure')) return 'Good';
  }
  if (key === 'contrast') {
    if (description.includes('low contrast')) return 'Poor';
    if (description.includes('medium') || description.includes('balanced')) return 'Good';
    if (description.includes('high contrast')) return 'Fair';
  }
  if (key === 'noise') {
    if (description.includes('high noise')) return 'Poor';
    if (description.includes('medium noise')) return 'Fair';
    if (description.includes('low noise')) return 'Good';
  }
  if (key === 'composition') {
    if (description.includes('poor')) return 'Poor';
    if (description.includes('average') || description.includes('neutral')) return 'Fair';
    if (description.includes('good')) return 'Good';
  }
  if (key === 'face_visibility') {
    if (category.toLowerCase() !== 'portrait photography') return 'Not assessed';
    if (description.startsWith('poor')) return 'Poor';
    if (description.startsWith('acceptable')) return 'Fair';
    if (description.startsWith('clear')) return 'Good';
  }
  if (key === 'object_visibility') {
    if (description.includes('not assessed') || item?.object_detected === false) return 'Not assessed';
    if (description.includes('unclear') || description.includes('clipped')) return 'Poor';
    if (description.includes('partly') || description.includes('weakly')) return 'Fair';
    if (description.includes('distinct') || description.includes('clearly')) return 'Good';
  }
  return 'Fair';
}

function metricExplanation(key, item, status, category) {
  const memberships = item?.memberships ?? {};
  if (key === 'blur') {
    const hasMemberships = Object.keys(memberships).length > 0;
    if (hasMemberships && memberships.High > memberships.Medium && memberships.High > memberships.Low) {
      return 'Significant blur is visible and fine details may be difficult to recover.';
    }
    if (hasMemberships && memberships.Medium > memberships.Low) {
      return 'Some softness is visible; a little sharpening may help.';
    }
    if (!hasMemberships && status === 'Poor') {
      return 'Significant blur is visible and fine details may be difficult to recover.';
    }
    if (!hasMemberships && status === 'Fair') return 'Some softness is visible; a little sharpening may help.';
    return item?.focus_discrepancy
      ? 'The background is sharper than the main subject, suggesting focus may have missed.'
      : 'The photo’s details look clear and in focus.';
  }
  if (key === 'brightness') {
    const hasMemberships = Object.keys(memberships).length > 0;
    const description = String(item?.description ?? '').toLowerCase();
    if (
      (hasMemberships && memberships.Dark > memberships.Normal && memberships.Dark >= memberships.Overexposed) ||
      (!hasMemberships && (description.includes('underexposed') || description.startsWith('dark')))
    ) {
      return 'The photo looks underexposed, with darker areas that may hide detail.';
    }
    if (
      (hasMemberships && memberships.Overexposed > memberships.Normal) ||
      (!hasMemberships && (description.includes('overexposed') || description.includes('blown')))
    ) {
      return 'The photo looks very bright, so some highlight detail may be lost.';
    }
    if (!hasMemberships && status === 'Poor') return 'Lighting falls outside the balanced range for this image.';
    return 'Lighting appears balanced, with detail visible across the image.';
  }
  if (key === 'contrast') {
    const hasMemberships = Object.keys(memberships).length > 0;
    if (
      (hasMemberships && memberships.Low > memberships.Medium && memberships.Low >= memberships.High) ||
      (!hasMemberships && status === 'Poor')
    ) {
      return 'The tones look quite similar, which can make the photo appear a little flat.';
    }
    if (
      (hasMemberships && memberships.High > memberships.Medium) ||
      (!hasMemberships && status === 'Fair' && Number(item?.value) > 68)
    ) {
      return 'The tones are strongly separated; this can add punch, depending on the scene.';
    }
    return 'The image has a natural range of light and dark tones.';
  }
  if (key === 'noise') {
    const hasMemberships = Object.keys(memberships).length > 0;
    if (
      (hasMemberships && memberships.High > memberships.Medium && memberships.High > memberships.Low) ||
      (!hasMemberships && status === 'Poor')
    ) {
      return 'Noticeable image grain or digital noise may distract from fine detail.';
    }
    if (hasMemberships && memberships.Medium > memberships.Low) {
      return 'Some image grain may be visible, especially in smooth or dark areas.';
    }
    if (!hasMemberships && status === 'Fair') return 'Some image grain may be visible, especially in smooth or dark areas.';
    return 'The image looks clean, with little visible grain.';
  }
  if (key === 'face_visibility') {
    if (category.toLowerCase() !== 'portrait photography') {
      return `Face detection is not used for ${category || 'this category'} and does not affect its assessment.`;
    }
    const count = Number(item?.face_count) || 0;
    if (count === 0) return 'No face was detected in this portrait-focused assessment.';
    const noun = count === 1 ? 'face' : 'faces';
    if (status === 'Poor') return `${count} ${noun} detected, but facial detail may be hard to make out.`;
    if (status === 'Fair') return `${count} ${noun} detected; visibility is acceptable but could be clearer.`;
    return `${count} ${noun} detected with clear facial detail.`;
  }
  if (key === 'object_visibility') {
    if (item?.object_detected === false) {
      return 'The foreground boundary was uncertain, so object visibility was left neutral rather than treated as a defect.';
    }
    if (status === 'Poor') return 'The primary object may be clipped or difficult to distinguish from its surroundings.';
    if (status === 'Fair') return 'The item is visible, though its framing or separation from the background could improve.';
    return 'A distinct primary object region is visible with comfortable framing.';
  }
  if (key === 'composition') {
    if (status === 'Poor') return 'The main visual area may be too close to an edge or awkwardly framed.';
    if (status === 'Fair') return 'The framing is acceptable, with some room to improve balance.';
    if (category.toLowerCase() === 'landscape photography') return 'The scene is framed with a balanced distribution of visual detail.';
    if (category.toLowerCase() === 'item photography') return 'The primary item has room around its detected frame.';
    if (category.toLowerCase() === 'portrait photography') return 'The portrait subject has balanced placement and safe framing.';
    return 'The image is framed with a balanced placement of its main visual area.';
  }
  return status === 'Good' ? 'This part of the image looks good.' : 'This part of the image may benefit from attention.';
}

export function getRecommendationPresentation(recommendation) {
  return recommendationCopy[recommendation] ?? {
    title: 'Review your photo',
    description: 'The photo was assessed, but no recommendation is available.',
    className: 'recover',
  };
}

export function buildMetricPresentations(features = {}, memberships = {}, category = 'General Photography') {
  const categoryName = category || 'General Photography';
  const categoryKey = categoryName.toLowerCase();
  return Object.entries(metricDefinitions)
    .filter(([key]) => {
      if (!features[key]) return false;
      if (key === 'face_visibility') return categoryKey === 'portrait photography';
      if (key === 'object_visibility') return categoryKey === 'item photography';
      return true;
    })
    .map(([key, definition]) => {
      const item = features[key];
      const notAssessed =
        (key === 'face_visibility' && item.applicable_to_quality === false) ||
        (key === 'object_visibility' && (item.applicable_to_quality === false || item.object_detected === false));
      const activeTerm = strongestTerm(memberships[key]);
      const status = notAssessed
        ? 'Not assessed'
        : definition.terms[activeTerm] ?? fallbackStatus(key, item, categoryName);
      return {
        key,
        label: definition.label,
        status,
        statusClass: statusClass(status),
        explanation: notAssessed && key === 'object_visibility'
          ? 'The foreground boundary was uncertain, so object visibility was left neutral rather than treated as a defect.'
          : metricExplanation(key, { ...item, memberships: memberships[key] ?? {} }, status, categoryName),
        item: { ...item, memberships: memberships[key] ?? {} },
        notAssessed,
      };
    });
}

export function buildPersonalizedFeedback(metrics, recommendation = 'KEEP') {
  const byKey = Object.fromEntries(metrics.map((metric) => [metric.key, metric]));
  const feedback = [];
  const add = (key, title, description, tone = 'attention') => {
    const metric = byKey[key];
    if (metric && !metric.notAssessed) feedback.push({ key, title, description, tone });
  };

  if (byKey.blur?.status === 'Poor') {
    add('blur', 'Blur detected', 'A steadier camera or faster shutter may help capture sharper detail.');
  } else if (byKey.blur?.status === 'Fair') {
    add('blur', 'A little softness', 'Try refining focus before taking the next shot.');
  }
  if (byKey.brightness?.status === 'Poor') {
    const isDark = byKey.brightness.item?.memberships?.Dark > byKey.brightness.item?.memberships?.Overexposed;
    add('brightness', isDark ? 'Lighting is low' : 'Highlights are very bright',
      isDark ? 'Add light or increase exposure to reveal more shadow detail.' : 'Reduce exposure or avoid direct glare to retain highlight detail.');
  }
  if (byKey.noise?.status === 'Poor' || byKey.noise?.status === 'Fair') {
    add('noise', byKey.noise.status === 'Poor' ? 'Visible image grain' : 'Some image grain',
      'More light or a lower camera sensitivity setting may help reduce grain.');
  }
  if (byKey.contrast?.status === 'Poor') {
    add('contrast', 'Low image contrast', 'Adjust tonal contrast gently while editing, if a punchier look suits the scene.');
  }
  if (byKey.composition?.status === 'Poor' || byKey.composition?.status === 'Fair') {
    add('composition', 'Framing could improve', 'Try leaving a little more space around the main visual area.');
  }
  if (byKey.face_visibility?.status === 'Poor') {
    add('face_visibility', 'Face detail is limited', 'For a portrait, move closer and focus on the person’s face.');
  }
  if (byKey.object_visibility?.status === 'Poor') {
    add('object_visibility', 'The item is hard to distinguish', 'Separate the item from nearby clutter and keep its edges inside the frame.');
  }

  if (feedback.length > 0) return feedback.slice(0, 3);
  if (recommendation !== 'KEEP') {
    return [{
      key: 'overall',
      title: 'The full image matters too',
      description: 'Some individual details look good, while the overall assessment considers the photo as a whole.',
      tone: 'neutral',
    }];
  }
  for (const key of ['blur', 'brightness', 'composition']) {
    if (byKey[key]?.status === 'Good' && !byKey[key].notAssessed) {
      feedback.push({
        key,
        title: key === 'blur' ? 'Clear details' : key === 'brightness' ? 'Balanced lighting' : 'Balanced framing',
        description: byKey[key].explanation,
        tone: 'positive',
      });
    }
    if (feedback.length === 2) break;
  }
  return feedback;
}
