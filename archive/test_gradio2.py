import urllib.request
import urllib.error
try:
    req = urllib.request.Request('https://krishna1205-face-analysis.hf.space/gradio_api/info')
    response = urllib.request.urlopen(req)
    print(response.read().decode())
except urllib.error.HTTPError as e:
    print(e.read().decode())
except Exception as e:
    print(str(e))
