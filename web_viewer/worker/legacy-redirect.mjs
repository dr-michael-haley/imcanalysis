// The previous application origin remains online solely for existing links.
export default {
  fetch(request) {
    if (!['GET', 'HEAD'].includes(request.method)) {
      return new Response('Method not allowed', {
        status: 405,
        headers: { Allow: 'GET, HEAD' },
      });
    }
    const destination = new URL(request.url);
    destination.protocol = 'https:';
    destination.hostname = 'imc.spatialbiology.app';
    destination.port = '';
    return new Response(null, {
      status: 308,
      headers: {
        Location: destination.href,
        'Cache-Control': 'public, max-age=300',
        'Referrer-Policy': 'no-referrer',
      },
    });
  },
};
