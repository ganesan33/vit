// Charts
let pieChart, barChart, lineChart;
// Map
let map, markerLayer;

// Setup map
function initMap() {
  map = L.map('map').setView([20, 0], 2.2);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18, attribution: '&copy; OpenStreetMap'
  }).addTo(map);
  markerLayer = L.layerGroup().addTo(map);
}

function resetMapMarkers() {
  markerLayer.clearLayers();
}

// Utility: group by key
function groupBy(arr, keyFn) {
  return arr.reduce((acc, item) => {
    const k = keyFn(item);
    acc[k] = acc[k] || [];
    acc[k].push(item);
    return acc;
  }, {});
}

// Utility: format YYYY-MM-DD from UTC string
function toDateOnly(ts) {
  if (!ts) return '';
  return ts.slice(0,10);
}

// Populate table
function renderTable(posts) {
  const tbody = document.getElementById('postsBody');
  tbody.innerHTML = '';
  posts.sort((a,b) => (a.created_utc < b.created_utc ? 1 : -1));
  for (const p of posts) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${p.created_utc || ''}</td>
      <td>${p.subreddit || ''}</td>
      <td title="${(p.body||'').replace(/"/g,'&quot;')}">${p.title || ''}</td>
      <td>${p.sentiment_label || ''} (${(p.sentiment_compound||0).toFixed(2)})</td>
      <td>${p.urgent_flag ? '🚨' : ''}</td>
      <td>${p.score ?? ''}</td>
      <td>${p.num_comments ?? ''}</td>
      <td><a href="${p.url}" target="_blank" rel="noopener">Open</a></td>
    `;
    tbody.appendChild(tr);
  }
}

// Summary list
function renderSummary(points) {
  const ul = document.getElementById('summaryList');
  ul.innerHTML = '';
  (points || []).forEach(pt => {
    const li = document.createElement('li');
    li.textContent = pt;
    ul.appendChild(li);
  });
}

// Stats boxes
function renderStats(total, neg, urgent) {
  document.getElementById('totalRecords').textContent = total;
  document.getElementById('negCount').textContent = neg;
  document.getElementById('urgentCount').textContent = urgent;
}

// Charts
function renderCharts(posts) {
  // Sentiment distribution
  const counts = { positive: 0, neutral: 0, negative: 0 };
  posts.forEach(p => {
    counts[p.sentiment_label] = (counts[p.sentiment_label] || 0) + 1;
  });

  const pieCtx = document.getElementById('sentimentPie').getContext('2d');
  if (pieChart) pieChart.destroy();
  pieChart = new Chart(pieCtx, {
    type: 'doughnut',
    data: {
      labels: ['Positive', 'Neutral', 'Negative'],
      datasets: [{
        data: [counts.positive||0, counts.neutral||0, counts.negative||0]
      }]
    },
    options: {
      plugins: {
        legend: { position: 'bottom' }
      }
    }
  });

  // Mentions by subreddit
  const bySub = groupBy(posts, p => p.subreddit || 'unknown');
  const subs = Object.keys(bySub);
  const subCounts = subs.map(s => bySub[s].length);
  const barCtx = document.getElementById('subredditBar').getContext('2d');
  if (barChart) barChart.destroy();
  barChart = new Chart(barCtx, {
    type: 'bar',
    data: { labels: subs, datasets: [{ label: 'Mentions', data: subCounts }] },
    options: {
      scales: { x: { ticks: { autoSkip: false, maxRotation: 60, minRotation: 0 } } },
      plugins: { legend: { display: false } }
    }
  });

  // Sentiment over time (avg compound per day)
  const byDay = groupBy(posts, p => toDateOnly(p.created_utc));
  const days = Object.keys(byDay).sort();
  const avgCompound = days.map(d => {
    const arr = byDay[d];
    const avg = arr.reduce((s, x) => s + (x.sentiment_compound || 0), 0) / arr.length;
    return +avg.toFixed(3);
  });
  const lineCtx = document.getElementById('timeLine').getContext('2d');
  if (lineChart) lineChart.destroy();
  lineChart = new Chart(lineCtx, {
    type: 'line',
    data: { labels: days, datasets: [{ label: 'Avg Sentiment (compound)', data: avgCompound, tension: 0.25 }] },
    options: {
      plugins: { legend: { position: 'bottom' } },
      scales: { y: { suggestedMin: -1, suggestedMax: 1 } }
    }
  });
}

// Map markers
function renderMap(posts) {
  resetMapMarkers();
  const coords = [];
  posts.forEach(p => {
    if (p.lat && p.lon) {
      const m = L.marker([p.lat, p.lon]).addTo(markerLayer);
      m.bindPopup(`
        <strong>${p.subreddit || ''}</strong><br/>
        ${p.title || ''}<br/>
        Sentiment: <em>${p.sentiment_label}</em> (${(p.sentiment_compound||0).toFixed(2)})
      `);
      coords.push([p.lat, p.lon]);
    }
  });
  if (coords.length) {
    const bounds = L.latLngBounds(coords);
    map.fitBounds(bounds.pad(0.3));
  } else {
    map.setView([20, 0], 2.2);
  }
}

// Alert logic
function renderAlert(signals) {
  const box = document.getElementById('alertBox');
  if (!signals) {
    box.className = 'alert hide';
    return;
  }
  const { urgent_count, negative_count, crisis_flag } = signals;
  if (crisis_flag) {
    box.textContent = `CRISIS SIGNAL: ${urgent_count} urgent, ${negative_count} negative.`;
    box.className = 'alert danger';
  } else if (urgent_count > 0 || negative_count > 0) {
    box.textContent = `Heads up: ${urgent_count} urgent and ${negative_count} negative mentions detected.`;
    box.className = 'alert warn';
  } else {
    box.textContent = 'All clear: no unusual spikes detected.';
    box.className = 'alert ok';
  }
}

// Form handling
async function runQuery(e) {
  e.preventDefault();
  const brands = document.getElementById('brands').value.split(',').map(s => s.trim()).filter(Boolean);
  const subsRaw = document.getElementById('subs').value.trim();
  const subreddits = subsRaw ? subsRaw.split(',').map(s => s.trim()).filter(Boolean) : ['all'];
  const timeFilter = document.getElementById('timeFilter').value;
  const limit = parseInt(document.getElementById('limit').value || '15', 10);
  const summarize = document.getElementById('summarize').value === 'true';

  const payload = { brands, subreddits, time_filter: timeFilter, limit, summarize };

  // UX: disable while loading
  const btn = e.target.querySelector('button[type="submit"]');
  const originalText = btn.textContent;
  btn.textContent = 'Fetching...';
  btn.disabled = true;

  try {
    const res = await fetch('/scrape', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Request failed');
    const data = await res.json();

    const posts = data.data || [];
    renderStats(data.total_records || 0,
                posts.filter(p => p.sentiment_label === 'negative').length,
                posts.filter(p => p.urgent_flag).length);

    renderSummary(data.summary || []);
    renderTable(posts);
    renderCharts(posts);
    renderMap(posts);
    renderAlert(data.signals);
  } catch (err) {
    console.error(err);
    renderAlert({ urgent_count: 0, negative_count: 0, crisis_flag: false });
    alert('Failed to fetch data. Check server logs/credentials.');
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initMap();
  document.getElementById('queryForm').addEventListener('submit', runQuery);

  // sensible defaults for quick test
  document.getElementById('brands').value = 'Nike, Adidas';
  document.getElementById('subs').value = 'all, Sneakers';
});
