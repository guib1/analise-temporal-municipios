__author__ = "Jan Urbansky"

# TODO: Change and describe structure of the links that have to be provided.
# TODO: Proper readme with examples.

from multiprocessing.dummy import Pool as Threadpool
import requests
import logging
import yaml
import os
import urllib.response
from http import cookiejar
import urllib.error
import urllib.request
import re

log = logging.getLogger('opendap_download')


class AuthenticationError(Exception):
    """Raised when NASA Earthdata credentials are invalid or expired."""
    pass


class DownloadError(Exception):
    """Raised when a file download fails (HTTP error, corrupted file, etc.)."""
    pass




class DownloadManager(object):
    __AUTHENTICATION_URL = 'https://urs.earthdata.nasa.gov/oauth/authorize'
    __username = ''
    __password = ''
    __download_urls = []
    __download_path = ''
    _authenticated_session = None

    def __init__(self, username='', password='', links=None, download_path='download'):
        self.set_username_and_password(username, password)
        self.download_urls = links
        self.download_path = download_path

        if logging.getLogger().getEffectiveLevel() == logging.INFO:
            logging.getLogger("requests").setLevel(logging.CRITICAL)
            logging.getLogger("urllib3").setLevel(logging.CRITICAL)
        
        log.debug('Init DownloadManager')

    @property
    def download_urls(self):
        return self.__download_urls

    @download_urls.setter
    def download_urls(self, links):
        """
        Setter for the links to download. The links have to be an array containing the URLs. The module will
        figure out the filename from the url and save it to the folder provided with download_path()
        :param links: The links to download
        :type links: List[str]
        """
        # TODO: Check if links have the right structure? Read filename from links?
        # Check if all links are formed properly
        if links is None:
            self.__download_urls = []
        else:
            for item in links:
                try:
                    self.get_filename(item)
                except AttributeError:
                    raise ValueError('The URL seems to not have the right structure: ', item)
            self.__download_urls = links

    @property
    def download_path(self):
        return self.__download_path

    @download_path.setter
    def download_path(self, file_path):
        self.__download_path = file_path

    def set_username_and_password(self, username, password):
        self.__username = username
        self.__password = password

    def read_credentials_from_yaml(self, file_path_to_yaml):
        with open(file_path_to_yaml, 'r') as f:
            credentials = yaml.load(f)
            log.debug('Credentials: ' + str(credentials))
            self.set_username_and_password(credentials['username'], credentials['password'])

    def _mp_download_wrapper(self, url_item):
        """
        Wrapper for parallel download. The function name cannot start with __ due to visibility issues.
        :param url_item:
        :type url_item:
        :return:
        :rtype:
        """
        query = url_item
        file_path = os.path.join(self.download_path, self.get_filename(query))
        self.__download_and_save_file(query, file_path)

    def _validate_authentication(self):
        """
        Pre-check authentication by requesting the first URL and verifying
        we get valid data (not an HTML error page or 401).
        """
        if not self.download_urls:
            raise DownloadError('No download URLs provided.')

        test_url = self.download_urls[0]
        r = self._authenticated_session.get(test_url, stream=True, timeout=60)

        if r.status_code == 401:
            raise AuthenticationError(
                'NASA Earthdata authentication failed (HTTP 401). '
                'Please verify your NASA_USER and NASA_PASSWORD in the .env file, '
                'and ensure GES DISC is authorized at '
                'https://urs.earthdata.nasa.gov (Profile → Applications → Authorize).'
            )
        if r.status_code == 403:
            raise AuthenticationError(
                'NASA Earthdata access forbidden (HTTP 403). '
                'Your account may not have GES DISC data access authorized. '
                'Visit https://urs.earthdata.nasa.gov → Profile → Applications → Authorize.'
            )
        if r.status_code != 200:
            raise DownloadError(
                f'Unexpected HTTP status {r.status_code} when accessing MERRA-2 data. '
                f'URL: {test_url[:120]}...'
            )

        # Check if response is HTML (error page) instead of binary data
        content_type = r.headers.get('Content-Type', '')
        first_bytes = r.content[:100]
        if b'<html' in first_bytes.lower() or b'<!doctype' in first_bytes.lower():
            raise AuthenticationError(
                'NASA Earthdata returned an HTML page instead of data. '
                'This usually means authentication failed silently. '
                'Please verify your credentials and GES DISC authorization.'
            )

        log.info('Authentication validated successfully.')
        r.close()

    def start_download(self, nr_of_threads=4):
        if self._authenticated_session is None:
            self._authenticated_session = self.__create_authenticated_sesseion()

        # Validate authentication before downloading hundreds of files
        self._validate_authentication()

        # Create the download folder.
        os.makedirs(self.download_path, exist_ok=True)
        # p = multiprocessing.Pool(nr_of_processes)
        p = Threadpool(nr_of_threads)
        p.map(self._mp_download_wrapper, self.download_urls)
        p.close()
        p.join()

    @staticmethod
    def get_filename(url):
        """
        Extracts the filename from the url. This method can also be used to check
        if the links have the correct structure
        :param url: The MERRA URL
        :type url: str
        :return: The filename
        :rtype: str
        """
        # Extract everything between a leading / and .nc4? . The problem with using this without any
        # other classification is, that the URLs have multiple / in their structure. The expressions [^/]* matches
        # everything but /. Combined with the outer expressions, this only matches the part between the last / and .nc4?
        reg_exp = r'(?<=/)[^/]*(?=.nc4?)'
        file_name = re.search(reg_exp, url).group(0)
        return file_name

    def __download_and_save_file(self, url, file_path):
        # Add timeout to prevent hanging on network drop
        r = self._authenticated_session.get(url, stream=True, timeout=60)

        if r.status_code == 401:
            raise AuthenticationError(
                f'Authentication failed (HTTP 401) for: {os.path.basename(file_path)}'
            )
        if r.status_code != 200:
            log.warning(
                'HTTP %d for %s — skipping file.',
                r.status_code, os.path.basename(file_path)
            )
            return r.status_code

        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        # Post-download validation: detect corrupted files (HTML error pages, etc.)
        file_size = os.path.getsize(file_path)
        if file_size < 100:  # A valid nc4 file is always > 100 bytes
            with open(file_path, 'rb') as f:
                content = f.read()
            if b'Access denied' in content or b'<html' in content.lower():
                os.remove(file_path)
                raise AuthenticationError(
                    f'Downloaded file for {os.path.basename(file_path)} '
                    f'contains an error response ({file_size} bytes): '
                    f'{content[:80].decode("utf-8", errors="replace")}'
                )

        return r.status_code

    def __create_authenticated_sesseion(self):
        s = requests.Session()
        s.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.85 Safari/537.36'}
        s.auth = (self.__username, self.__password)
        s.cookies = self.__authorize_cookies_with_urllib()

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            r = s.get(self.download_urls[0])
            log.debug('Authentication Status')
            log.debug(r.status_code)
            log.debug(r.headers)
            log.debug(r.cookies)

            log.debug('Sessions Data')
            log.debug(s.cookies)
            log.debug(s.headers)
        return s

    def __authorize_cookies_with_urllib(self):
        username = self.__username
        password = self.__password
        top_level_url = "https://urs.earthdata.nasa.gov"

        # create an authorization handler
        p = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        p.add_password(None, top_level_url, username, password);

        auth_handler = urllib.request.HTTPBasicAuthHandler(p)
        auth_cookie_jar = cookiejar.CookieJar()
        cookie_jar = urllib.request.HTTPCookieProcessor(auth_cookie_jar)
        opener = urllib.request.build_opener(auth_handler, cookie_jar)

        urllib.request.install_opener(opener)

        try:
            # The merra portal moved the authentication to the download level. Before this change you had to
            # provide username and password on the overview page. For example:
            # goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/
            # authentication_url = 'https://goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/1980/01/MERRA2_100.tavg1_2d_slv_Nx.19800101.nc4.ascii?U2M[0:1:1][0:1:1][0:1:1]'
            # Changes:
            # Authenticate with the first url in the links.
            # Request the website and initialiaze the BasicAuth. This will populate the auth_cookie_jar
            authentication_url = self.download_urls[0]
            result = opener.open(authentication_url)
            log.debug(list(auth_cookie_jar))
            log.debug(list(auth_cookie_jar)[0])
            log.debug(list(auth_cookie_jar)[1])

        except urllib.error.HTTPError:
            raise ValueError('Username and or Password are not correct!')
        except IOError as e:
            log.warning(e)
            raise IOError
        except IndexError as e:
            log.warning(e)
            raise IndexError('download_urls is not set')

        return auth_cookie_jar


if __name__ == '__main__':
    link = [
        'http://goldsmr4.sci.gsfc.nasa.gov:80/opendap/MERRA2/M2T1NXSLV.5.12.4/2014/01/MERRA2_400.tavg1_2d_slv_Nx.20140101.nc4.nc4?U2M[0:1:5][358:1:360][573:1:575],U10M[0:1:5][358:1:360][573:1:575],U50M[0:1:5][358:1:360][573:1:575],V2M[0:1:5][358:1:360][573:1:575],V10M[0:1:5][358:1:360][573:1:575],V50M[0:1:5][358:1:360][573:1:575]']

    logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
    dl = DownloadManager()
    dl.download_path = 'downlaod123'
    dl.read_credentials_from_yaml((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'authentication.yaml')))
    dl.download_urls = link
    dl.start_download()

