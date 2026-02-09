import random
import re
import requests as rq
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fake_useragent import UserAgent
from pathlib import Path
from requests.exceptions import RequestException, Timeout, HTTPError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
from time import sleep
from tqdm import tqdm
import os
status_code = defaultdict(lambda: "Not define in table")
status_code[202] = "202: Accepted"
status_code[204] = "204: No Content"
status_code[400] = "400: Bad Request"
status_code[401] = "401: Unauthorized"
status_code[403] = "403: Forbidden"
status_code[404] = "404: Not Found"
status_code[409] = "409: Conflict"
status_code[429] = "429: Too Many Requests"


# CURRENT_PATH=r"D:\MML\AIDCC\dataset\fiveK\C"#本文件所在路径
# os.chdir(CURRENT_PATH)#
# https://data.csail.mit.edu/graphics/fivek/img/tiff16_<a b c d e>/*.tif


class FivekCrawler:
    """Crawl the fivek experts.

    Arguments
    ---------
    expert_list : list
        a list contain available experts id, ["a", "b", "c", "d", "e"]
    max_workers : int
        how many images downloaded at the same time.
    saving_dir : str
        the directory path for saving the edited photo,
        default is under current directory.
    num_images : int
        how many images to download for each expert. The maximum value is
        5000, which is also the default.
    """

    def __init__(
        self,
        expert_list,
        max_workers,
        # saving_dir=r"D:\MML\AIDCC\dataset\fiveK\C",
        saving_dir = None,
        image_from: int = 0,
        image_to: int = 5000,
    ):
        self.expert_list = expert_list
        self.max_workers = max_workers
        self.saving_dir = saving_dir
        self.image_from = image_from
        self.image_to = image_to
        self.incomplete_images = []
        self.fivek_src = "https://data.csail.mit.edu/graphics/fivek"
        self._local = threading.local()
        # create one UserAgent instance to avoid repeated initialization
        try:
            self.ua = UserAgent()
        except Exception:
            self.ua = None

        assert image_from < image_to, "image_to must larger than image_from."
        assert 0 <= image_from <= 5000, "image_from is out of range."
        assert 0 <= image_to <= 5000, "image_from is out of range."

        if saving_dir is None:
            self.saving_dir = Path(__file__).parent.resolve()

    def _create_expert_folder(self, expert_id, saving_dir=None):
        """Create the folder for saving images edited by each expert.

        Arguments
        ---------
        expert_id : str
            an expert id in [a, b, c, d, e]
        saving_dir : str
            the directory path for saving

        Returns
        -------
        folder_dir : Path
        """

        folder_dir = Path(f"{saving_dir}/fivek_expert/tiff16_{expert_id}")
        try:
            folder_dir.mkdir(parents=True)
        except FileExistsError:
            print(f"target directory ({folder_dir}) already exists.")

        return folder_dir

    def _choose_header(self):
        """Retrun a random User-Agent."""
        if self.ua:
            return {"User-Agent": self.ua.random}
        return {"User-Agent": "Mozilla/5.0 (compatible)"}
        # return {"User-Agent": user_agent().random}

    def _get_session(self):
        """Return a thread-local requests.Session configured with a retry adapter."""
        if getattr(self._local, "session", None) is None:
            session = rq.Session()
            retries = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=["HEAD", "GET", "OPTIONS"],
            )
            adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._local.session = session
        return self._local.session

    def _make_request(self, header, timeout):
        """Send a request to ask for the web html.

        Arguments
        ---------
        header : dict
            User-Agent
        timeout : int
            if not reply within the specified time, it will raise the exception.

        Returns
        -------
        html : str
            web html
        """
        try:
            # small randomized pause to avoid hammering the host
            sleep(random.random() * 0.2)
            web_rq = self._get_session().get(url=self.fivek_src, headers=header, timeout=timeout)
        except RequestException as err:
            print(err.__class__.__name__)  # debug
        except Exception as err:
            print(err.__class__.__name__)  # debug
        else:
            if web_rq.status_code == 200:
                print("request successed")
                return web_rq.text
            else:
                print(f"Status_code: {status_code[web_rq.status_code]}")  # debug
                exit()



    def _get_expert_images_url(self, web_html):
        """Select the images URL for the specific experts

        Arguments
        ---------
        web_html : str
            the html of fivek dataset website.
        """
        # https://stackoverflow.com/a/71370571
        expert_list = "".join(self.expert_list)
        urls = re.finditer(
            rf"img/tiff16_[{expert_list}]/\S*.tif",
            web_html,
        )
        for url in urls:
            url_index = int(url.group().split("/")[-1][1:5])
            if self.image_from <= url_index <= self.image_to:
                yield url.group()

    def download_image(self, url, image_path):
        """Download the image.

        Arguments
        ---------
        url : str
            the link to the image witch format is img/tiff16_{expert_id}/*.tif
        image_path : str
            the current saving path for an image.
        """

        retry = 0
        chunk_size = 64 * 1024
        while retry < 3:
            try:
                session = self._get_session()
                with session.get(
                    url=f"{self.fivek_src}/{url}",
                    headers=self._choose_header(),
                    stream=True,
                    timeout=(5, 30),
                ) as r:
                    r.raise_for_status()
                    image_path = Path(image_path)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(image_path, "wb") as f:
                        for data in r.iter_content(chunk_size):
                            if data:
                                f.write(data)
                # successful, exit retry loop
                break
            except Timeout:
                retry += 1
                if retry == 3:
                    print(f"\r{Path(url).name:>7s} : [Timeout] Retry({retry})")
                    self.incomplete_images.append([url, str(image_path)])
                else:
                    print(f"\r{Path(url).name:>7s} : [Timeout] Retry({retry})", end="")
                    sleep(0.5 * (2 ** retry))
            except (HTTPError, RequestException) as err:
                print(f"{Path(url).name:>7s} : {err}")
                self.incomplete_images.append([url, str(image_path)])
                break

    def main(self):
        html = self._make_request(self._choose_header(), timeout=5)
        urls = self._get_expert_images_url(html)

        # ensure expert folders exist
        for i in self.expert_list:
            self._create_expert_folder(i, self.saving_dir)

        total_images = (self.image_to - self.image_from) * len(self.expert_list)

        with tqdm(total=total_images) as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as saver:
                futures = []
                for i, url in enumerate(urls, start=1):
                    # determine expert id from url and build local path
                    m = re.search(r"tiff16_([a-zA-Z])", url)
                    expert_id = m.group(1) if m else self.expert_list[0]
                    image_path = Path(self.saving_dir) / "fivek_expert" / f"tiff16_{expert_id}" / Path(url).name
                    future = saver.submit(self.download_image, url, str(image_path))
                    futures.append(future)
                    if i % total_images == 0:
                        break
                for future in as_completed(futures):
                    pbar.update(1)

        remains = self.incomplete_images
        self.incomplete_images = []
        while remains:
            with tqdm(total=int(len(remains))) as pbar:
                with ThreadPoolExecutor(max_workers=self.max_workers) as saver:
                    futures = []
                    for i_url, i_path in remains:
                        f = saver.submit(self.download_image, i_url, i_path)
                        futures.append(f)
                    for future in as_completed(futures):
                        pbar.update(1)
            remains = self.incomplete_images


if __name__ == "__main__":
    crawler = FivekCrawler(
        # expert_list=["a", "b", "c", "d", "e"],
        expert_list=["c"],
        max_workers=5,
        image_from=1500,
        image_to=3000,
    )

    try:
        crawler.main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")