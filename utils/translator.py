import json
from tencentcloud.common import credential  # 这里需要安装腾讯翻译sdk
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tmt.v20180321 import tmt_client, models
from params import tencent_info


class TencentTranslator:
    def __init__(self, SecretId=tencent_info['SecretId'], SecretKey=tencent_info['SecretKey']):
        self.cred = credential.Credential(SecretId, SecretKey)
        self.httpProfile = HttpProfile()
        self.httpProfile.endpoint = "tmt.tencentcloudapi.com"
        self.clientProfile = ClientProfile()
        self.clientProfile.httpProfile = self.httpProfile
        self.client = tmt_client.TmtClient(self.cred, "ap-beijing", self.clientProfile)

    def _trans(self, text, source='zh', target='en'):
        req = models.TextTranslateRequest()
        req.SourceText = text
        req.Source = source
        req.Target = target
        req.ProjectId = 0
        resp = self.client.TextTranslate(req)
        data = json.loads(resp.to_json_string())
        return data['TargetText']

    def translate(self, text, source='zh', target='en'):
        if isinstance(text, list):
            return self._trans('\n'.join(text), source, target).split('\n')

        elif isinstance(text, str):
            return self._trans(text, source, target)