// Keep bookmarked sections and query parameters when following the renamed guide.
const destination = new URL('imc_explorer.html', window.location.href);
destination.search = window.location.search;
destination.hash = window.location.hash;
window.location.replace(destination.href);
