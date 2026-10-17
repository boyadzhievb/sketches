(() => {
  const filterButtons = document.querySelectorAll('.filter-list button');
  const cards = document.querySelectorAll('.art-card');
  const countEl = document.querySelector('.result-count');
  const lightbox = document.getElementById('lightbox');
  const lightboxInner = lightbox.querySelector('.lightbox-inner');
  const closeBtn = lightbox.querySelector('.close-button');

  let activeCategory = 'all';

  function updateFilters() {
    let visible = 0;
    cards.forEach(card => {
      const cat = card.dataset.category;
      const show = activeCategory === 'all' || cat === activeCategory;
      card.hidden = !show;
      if (show) visible++;
    });
    countEl.textContent = String(visible).padStart(2, '0') + ' works';
    filterButtons.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.filter === activeCategory);
    });
  }

  filterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      activeCategory = btn.dataset.filter;
      updateFilters();
    });
  });

  function openLightbox(card) {
    const doodleId = card.dataset.doodleId;
    const title = card.dataset.title;
    const category = card.dataset.category;
    const year = card.dataset.year;
    const index = card.dataset.index;
    const photoSrc = card.dataset.photo;

    let paperHTML;
    if (photoSrc) {
      paperHTML = `<div class="lightbox-paper">
        <img class="lightbox-photo" src="${photoSrc}" alt="${title}">
      </div>`;
    } else {
      const svg = card.querySelector('.doodle');
      const svgClone = svg.cloneNode(true);
      paperHTML = `<div class="lightbox-paper">${svgClone.outerHTML}</div>`;
    }

    lightboxInner.innerHTML = `
      ${paperHTML}
      <div class="lightbox-caption">
        <span>${index}</span>
        <div>
          <h2>${title}</h2>
          <p>${category} · blue pen on paper · ${year}</p>
        </div>
      </div>
    `;

    lightbox.hidden = false;
    document.body.style.overflow = 'hidden';
    closeBtn.focus();
  }

  function closeLightbox() {
    lightbox.hidden = true;
    document.body.style.overflow = '';
  }

  cards.forEach(card => {
    card.addEventListener('click', () => openLightbox(card));
  });

  document.querySelectorAll('.sketch-sheet-card').forEach(card => {
    card.addEventListener('click', () => {
      const src = card.dataset.photo;
      const title = card.dataset.title;
      lightboxInner.innerHTML = `
        <div class="lightbox-paper" style="aspect-ratio: auto;">
          <img class="lightbox-photo" src="${src}" alt="${title}">
        </div>
        <div class="lightbox-caption">
          <span></span>
          <div>
            <h2>${title}</h2>
            <p>full sketch sheet · blue pen on A4 paper · 2026</p>
          </div>
        </div>
      `;
      lightbox.hidden = false;
      document.body.style.overflow = 'hidden';
      closeBtn.focus();
    });
  });

  closeBtn.addEventListener('click', closeLightbox);

  lightbox.addEventListener('mousedown', e => {
    if (e.target === lightbox) closeLightbox();
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !lightbox.hidden) closeLightbox();
  });

  updateFilters();

  // ─── Extracted Doodles ───

  const extractedGallery = document.getElementById('extracted-gallery');
  const sheetFilterContainer = document.querySelector('.doodle-sheet-filter');

  if (extractedGallery) {
    fetch('doodles.json')
      .then(r => r.json())
      .then(doodles => {
        // Build sheet filter buttons
        const sheets = [...new Set(doodles.map(d => d.sheet))];
        sheets.forEach(sheet => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.dataset.sheet = sheet;
          const count = doodles.filter(d => d.sheet === sheet).length;
          btn.textContent = sheet.replace('IMG_', '').replace('IMG_', '') + ` (${count})`;
          sheetFilterContainer.appendChild(btn);
        });

        let activeSheet = 'all';

        function renderDoodles() {
          extractedGallery.innerHTML = '';
          const visible = activeSheet === 'all'
            ? doodles
            : doodles.filter(d => d.sheet === activeSheet);

          visible.forEach((d, i) => {
            const card = document.createElement('button');
            card.className = 'extracted-card';
            card.type = 'button';
            card.setAttribute('aria-label', `View doodle ${d.sheet} #${d.index}`);
            card.innerHTML = `
              <img src="images/doodles/${d.file}" alt="Extracted doodle ${d.index} from ${d.sheet}" loading="lazy">
              <span class="card-label">${d.sheet.replace('IMG_', '')} #${d.index}</span>
            `;
            card.addEventListener('click', () => {
              lightboxInner.innerHTML = `
                <div class="lightbox-paper" style="aspect-ratio: auto; background: var(--card-bg); display: flex; align-items: center; justify-content: center; padding: 2rem;">
                  <img class="lightbox-photo" src="images/doodles/${d.file}" alt="Doodle ${d.index} from ${d.sheet}" style="max-height: 70vh; object-fit: contain;">
                </div>
                <div class="lightbox-caption">
                  <span>${String(i + 1).padStart(3, '0')}</span>
                  <div>
                    <h2>${d.sheet.replace('IMG_', 'Sheet ')} — Doodle ${d.index}</h2>
                    <p>extracted · blue pen on paper · 2026</p>
                  </div>
                </div>
              `;
              lightbox.hidden = false;
              document.body.style.overflow = 'hidden';
              closeBtn.focus();
            });
            extractedGallery.appendChild(card);
          });
        }

        // Sheet filter clicks
        sheetFilterContainer.addEventListener('click', e => {
          const btn = e.target.closest('button');
          if (!btn) return;
          activeSheet = btn.dataset.sheet;
          sheetFilterContainer.querySelectorAll('button').forEach(b =>
            b.classList.toggle('active', b.dataset.sheet === activeSheet)
          );
          renderDoodles();
        });

        renderDoodles();
      })
      .catch(err => {
        extractedGallery.innerHTML = '<p style="color: var(--text-dim); padding: 2rem;">Could not load doodles. Run the extraction script first.</p>';
      });
  }
})();
