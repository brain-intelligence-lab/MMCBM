# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""
import os.path

import gradio as gr
import pandas as pd
import torch
from gradio import components

from web.intervention import Intervention

user_csv = 'CSV/human_evaluation/user.csv'
if not os.path.exists(user_csv):
    user_passwd = {
        'root': '1998',
        'liuy': '1234',
        'test': '1234',
    }
    pd.DataFrame(user_passwd.items(), columns=['user', 'passwd']).to_csv(user_csv, index=False)
else:
    user_passwd = pd.read_csv(user_csv)
    user_passwd = dict(zip(user_passwd['user'], user_passwd['passwd'].map(str)))

github = """
               <div style="display: flex; align-items: center; justify-content: center; height: 100px;">
                    <a href='https://github.com/ly1998117/MMCBM' target='_blank' style="text-decoration: none; color: black; text-align: center;">
                        <img src='https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png' width='50' height='50' style="display: block; margin: 0 auto;"/>
                        GitHub repository
                    </a>
                </div>
           """
texts = {
    'title': {'en': '# MMCBM_2 Interface', 'cn': '# 黑盒模型人工评估'},
    'desc': {
        'en': "### Prediction: Upload Fundus Images, Click Predict button to get the Top-10 concepts and prediction. \n"
              "### Intervention: After adjusting the sliders, "
              "click the 'Intervention' button to update the prediction.",
        'cn': '### 预测：单击一行跳转模型预测。\n'
              '### 评估：参考模型预测，给出你自己的预测。 注意：根据实验需求，部分病人的模型预测结果为随机值！'},
    'predict': {'en': 'Predict', 'cn': '预测'},
    'intervene': {'en': 'Intervene Concept', 'cn': '干预概念'},
    'report': {'en': 'Generate Report', 'cn': '生成报告'},
    'icon': {'en': github, 'cn': github},
}

fa_e_label = gr.HTML(
    value="<div style='position: relative;top: -2px;'>"
          "<span style='position: absolute; left: 0; top: 0; padding: 0px; font-size: 14px; color: #6b727f;'>FA-E</span>"
          "</div>")
fa_m_label = gr.HTML(
    value="<div style='position: relative;top: -2px'>"
          "<span style='position: absolute; left: 0; top: 0; padding: 0px; font-size: 14px; color: #6b727f;'>FA-M</span>"
          "</div>")
fa_l_label = gr.HTML(
    value="<div style='position: relative;top: -2px'>"
          "<span style='position: absolute; left: 0; top: 0; padding: 0px; font-size: 14px; color: #6b727f;'>FA-L</span>"
          "</div>")
icga_e_label = gr.HTML(
    value="<div style='position: relative;top: -2px'>"
          "<span style='position: absolute; left: 0; top: 0; padding: 0px; font-size: 14px; color: #6b727f;'>ICGA-E</span>"
          "</div>")
icga_m_label = gr.HTML(
    value="<div style='position: relative;top: -2px'>"
          "<span style='position: absolute; left: 0; top: 0; padding: 0px; font-size: 14px; color: #6b727f;'>ICGA-M</span>"
          "</div>")
icga_l_label = gr.HTML(
    value="<div style='position: relative;top: -2px'>"
          "<span style='position: absolute; left: 0; top: 0; padding: 0px; font-size: 14px; color: #6b727f;'>ICGA-L</span>"
          "</div>")
us_label = gr.HTML(
    value="<div style='position: relative;top: -2px'>"
          "<span style='position: absolute; left: 0; top: 0; padding: 0px; font-size: 14px; color: #6b727f;'>US</span>"
          "</div>")
# 3.47.1 tool=False
user = gr.Text(label='当前用户', min_width=2, scale=1, max_lines=1, interactive=False, text_align='center')
name = gr.Text(label='病人', min_width=1, scale=1, interactive=False)
pathology = gr.Text(label='疾病', min_width=1, scale=1, interactive=False)
diagnose = gr.Text(label='诊断', min_width=1, scale=1, interactive=False)
is_random = gr.Text(label='随机', min_width=1, scale=1, interactive=False, visible=False)

fa_e = gr.Image(type="pil", label='FA-早', min_width=2, scale=1, show_label=False,
                show_download_button=False, interactive=False)
fa_m = gr.Image(type="pil", label='FA-中', min_width=2, scale=1, show_label=False,
                show_download_button=False, interactive=False)
fa_l = gr.Image(type="pil", label='FA-晚', min_width=2, scale=1, show_label=False,
                show_download_button=False, interactive=False)
icga_e = gr.Image(type="pil", label='ICGA-早', min_width=2, scale=1, show_label=False,
                  show_download_button=False, interactive=False)
icga_m = gr.Image(type="pil", label='ICGA-中', min_width=2, scale=1, show_label=False,
                  show_download_button=False, interactive=False)
icga_l = gr.Image(type="pil", label='ICGA-晚', min_width=2, scale=1, show_label=False,
                  show_download_button=False, interactive=False)
us = gr.Image(type="pil", label='US', min_width=2, show_label=False, show_download_button=False, interactive=False)


# buttons


def update_texts(language):
    return [gr.Markdown(texts['title'][language]),
            gr.Markdown(texts['desc'][language])]


class EasyDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class UserSession:
    def __init__(self, data, users, default='root'):
        self.root = 'CSV/human_evaluation/BlackBox'
        os.makedirs(self.root, exist_ok=True)
        self.default = default
        self.current_user = self.default
        if default in users:
            users.remove(default)
        self.users = users
        self.data = self._load_data(data)
        self.state = {u: EasyDict() for u in users}
        self.state[default] = EasyDict()

    def save_data(self):
        cache_path = os.path.join(self.root, 'test.csv')
        self.data.to_csv(cache_path, index=False)

    def set_data(self, name, diagnose):
        self.data.loc[(self.data['index'] == name) & (self.data['user'] == self.current_user), 'human_pred'] = diagnose
        self.save_data()

    def get_data_from_name(self, name):
        return self.data.loc[(self.data['index'] == name) & (self.data['user'] == self.current_user)]

    def is_random_from_id(self, data_id, diagnosed=False):
        data = self._get_data(diagnosed)
        return data.iloc[data_id]['random']

    def _get_data(self, diagnosed=False):
        if diagnosed:
            data = self.data[self.data['human_pred'] != '待诊断']
        else:
            data = self.data[self.data['human_pred'] == '待诊断']
        if self.current_user != self.default:
            data = data[data['user'] == self.current_user]
        return data

    def get_data(self, diagnosed=False):
        def _fn(x):
            if self.current_user != self.default:
                path = [x['user'], x['index'], x['human_pred'], '****', '****']
            else:
                path = [x['user'], x['index'], x['human_pred'], x['pathology'], x['random']]

            [path.extend(x['path'][m]) for m in ['FA', 'ICGA', 'US']]
            return path

        data = self._get_data(diagnosed=diagnosed).apply(_fn, axis=1)
        if len(data) == 0:
            return []
        return data.to_list()

    def table_cut(self, length, labels, check_num=-1):
        labels = labels.copy()
        while check_num != -1 and len(labels) <= length // check_num:
            labels.append('')
        while check_num != -1 and len(labels) > length // check_num + 1:
            labels.pop()
        cut_list = [labels[i % len(labels)] for i in range(length)]
        return cut_list

    def _load_data(self, data):
        cache_path = os.path.join(self.root, 'test.csv')
        if os.path.exists(cache_path):
            from ast import literal_eval
            cache_path = pd.read_csv(cache_path)
            cache_path['path'] = cache_path['path'].map(literal_eval)
            return cache_path
        data = data.reset_index()
        data['index'] = data.index
        data['index'] = data['index'].map(lambda x: f'id_{x}')
        self.order = 1

        def _divide(x):
            x['user'] = self.table_cut(length=len(x), labels=self.users[::self.order])
            self.order *= -1
            return x

        def _divide_random(x):
            x['random'] = self.table_cut(length=len(x), labels=[True, False])
            return x

        data = data.groupby(by='pathology').apply(_divide).reset_index(drop=True)
        data = data.groupby(by='user').apply(_divide_random).reset_index(drop=True)
        data['human_pred'] = '待诊断'
        data.to_csv(cache_path, index=False)
        return data

    def divide_by_users(self):
        request = gr.Request()
        return request.username

    def __getattr__(self, item):
        # 检查是否试图访问的属性在用户特定的状态数据中
        if item in ['root', 'default', 'current_user', 'data', 'users',
                    '_load_data', 'table_cut', 'divide_by_users', 'state', 'order']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

        # 尝试从 self.state[user.value] 返回属性值
        try:
            return self.state[self.current_user][item]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object User {self.current_user} has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key in ['root', 'default', 'current_user', 'data', 'users', 'state', 'order',
                   '_load_data', 'table_cut', 'divide_by_users', 'set_data']:
            super().__setattr__(key, value)
        else:
            if self.current_user == self.default:
                for user in self.users:
                    self.state[user][key] = value
            self.state[self.current_user][key] = value

    def __getitem__(self, user):
        self.current_user = user
        return self


predict = Intervention(
    json_path='result/Efficientb0_SCLS_attnscls_CrossEntropy_32/fold_0',
    backbone='Efficientb0_SCLS_attnscls_CrossEntropy_32/fold_0',
    idx=180,
    device='cpu',
    normalize='linear',
)
session = UserSession(data=predict.get_test_data(mask=False, format='dataframe'), users=list(user_passwd.keys()))
session.language = "cn"
callback = gr.CSVLogger()


########################### Functions ###########################
def auth_fn(user, passwd):
    if user not in user_passwd.keys():
        return False
    return user_passwd[user] == passwd


def load_fn(request: gr.Request):
    return request.username


def load_param(user):
    return session[user].language, gr.Dataset(
        samples=session[user].get_data()), gr.Dataset(
        samples=session[user].get_data(diagnosed=True))


def dataset_fn(user, data_id):
    data = session[user].get_data()[data_id]
    return data[1], data[2], *data[5:], session[user].is_random_from_id(data_id)


def dataset2_fn(user, data_id):
    data = session[user].get_data(diagnosed=True)[data_id]
    return data[1], data[2], *data[5:], session[user].is_random_from_id(data_id, diagnosed=True)


def language_fn(user, x):
    session[user].language = x


def predict_label(user, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us, is_random):
    predict.set_random(eval(is_random))
    return predict.predict_label(session[user].language, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us)


def checkbox_fn(x):
    return gr.Text(value=x)


def diagnose_fn(user, name, diagnose, label, is_random):
    if name == '':
        gr.Warning("请在诊断栏选择一个病人")
        return gr.Radio(), None, None, gr.Tabs(), '', '待诊断'
    if diagnose == '待诊断':
        gr.Warning("请选择一个疾病")
        return gr.Radio(), None, None, gr.Tabs(), '', '待诊断'
    session[user].set_data(name, diagnose)
    gr.Info("诊断成功，移动到已诊断栏")
    if 'confidences' not in label:
        confidences = [dict(label=k, confidence=v) for k, v in label.items()]
        label = dict(label=sorted(label.items(), key=lambda x: x[1], reverse=True)[0][0], confidences=confidences)
    callback.flag([name, is_random, label], username=user,
                  flag_option=diagnose)
    return (gr.Radio(value=None), gr.Dataset(samples=session[user].get_data()),
            gr.Dataset(samples=session[user].get_data(diagnosed=True)), gr.Tabs(selected=0), '', '待诊断')


def grad_cam_fn(user, name):
    # cams = predict.grad_cam(fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us)
    # imgs = []
    # for modality in cams.keys():
    #     imgs.extend([gr.Image(value=i) for i in cams[modality]])
    data = session[user].get_data_from_name(name).iloc[0]['path']
    imgs = []
    for m in ['FA', 'ICGA', 'US']:
        imgs.extend([gr.Image(value=i.replace('data', 'cam')) for i in data[m]])
    return imgs


########################### Gradio ###########################
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=10):
            title = gr.Markdown(texts['title'][session.language])

        with gr.Column(scale=2):
            link = gr.Markdown(texts['icon'][session.language])

        with gr.Column(scale=1):
            user.render()

    with gr.Row():
        with gr.Column(scale=5):
            desc = gr.Markdown(texts['desc'][session.language])
        with gr.Column(scale=1, min_width=1):
            lan = gr.Dropdown(label="Language", value=session.language,
                              choices=["en", "cn"], elem_id="language",
                              multiselect=False,
                              min_width=1)
            is_random.render()

    with gr.Tabs() as tabs:
        with gr.TabItem("待诊断", id=0):
            dataset = gr.Dataset(
                samples=None,
                components=[gr.Text(label='用户', min_width=2, scale=1, visible=False), name, diagnose,
                            gr.Text(label='疾病', min_width=2, scale=1, visible=False),
                            gr.Text(label='随机预测', min_width=2, scale=1, visible=False),
                            fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us],
                type="index",
                label="点击一行，跳转到诊断界面",
                samples_per_page=5,
            )

        with gr.TabItem("已诊断", id=2):
            dataset2 = gr.Dataset(
                samples=None,
                components=[gr.Text(label='用户', min_width=2, scale=1, visible=False), name, diagnose,
                            gr.Text(label='疾病', min_width=2, scale=1, visible=False),
                            gr.Text(label='随机预测', min_width=2, scale=1, visible=False),
                            fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us],
                type="index",
                label="点击一行，跳转到诊断界面",
                samples_per_page=5,
            )

        with gr.TabItem("辅助诊断", id=1) as tag_diagnose:
            with gr.Row(equal_height=True):
                with gr.Column(scale=3, min_width=160):
                    with gr.Accordion("病人三模态图像", open=True, elem_id="input-panel"):
                        with gr.Row(equal_height=True):
                            name.render()
                            diagnose.render()
                        with gr.Row(equal_height=True):
                            fa_e_label.render()
                            fa_m_label.render()
                            fa_l_label.render()

                        with gr.Row(equal_height=True):
                            fa_e.render()
                            fa_m.render()
                            fa_l.render()
                        with gr.Row(equal_height=True):
                            icga_e_label.render()
                            icga_m_label.render()
                            icga_l_label.render()

                        with gr.Row(equal_height=True):
                            icga_e.render()
                            icga_m.render()
                            icga_l.render()
                        with gr.Row(equal_height=True):
                            us_label.render()
                        with gr.Row(equal_height=True):
                            us.render()

                with gr.Column(scale=3, min_width=160):
                    with gr.Accordion("模型预测", open=True, elem_id="output-panel"):
                        label = gr.Label(num_top_classes=3, label="模型预测")

                    with gr.Accordion("人工诊断", open=True, elem_id="human-panel"):
                        with gr.Row():
                            checkbox = gr.Radio(label='选择疾病，确保只选中一个疾病',
                                                choices=['血管瘤', '转移癌', '黑色素瘤'])
                        with gr.Row():
                            diagnose_btn = gr.Button(value="确定")
    ############################################## Trigger ##############################################
    demo.load(load_fn, inputs=None, outputs=user).then(load_param, inputs=user,
                                                       outputs=[lan, dataset, dataset2])

    callback.setup([name, is_random, label],
                   flagging_dir=f"{session.root}/flagged_backbone_data_points")
    lan.change(update_texts, inputs=lan,
               outputs=[title, desc]).then(
        fn=language_fn, inputs=[user, lan]
    )

    dataset.click(
        fn=dataset_fn,
        inputs=[user, dataset],
        outputs=[name, diagnose, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us, is_random]
    ).then(
        fn=lambda x: gr.Tabs(selected=1),
        inputs=None,
        outputs=tabs,
    ).then(
        fn=predict_label,
        inputs=[user, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us, is_random],
        outputs=label
    ).then(
        fn=grad_cam_fn,
        inputs=[user, name],
        outputs=[fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us],
    )

    dataset2.click(
        fn=dataset2_fn,
        inputs=[user, dataset2],
        outputs=[name, diagnose, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us, is_random]
    ).then(
        fn=lambda x: gr.Tabs(selected=1),
        inputs=None,
        outputs=tabs,
    ).then(
        fn=predict_label,
        inputs=[user, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us, is_random],
        outputs=label
    ).then(
        fn=grad_cam_fn,
        inputs=[user, name],
        outputs=[fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us],
    )

    checkbox.change(
        fn=checkbox_fn,
        inputs=checkbox,
        outputs=diagnose
    )

    diagnose_btn.click(
        fn=diagnose_fn,
        inputs=[user, name, diagnose, label, is_random],
        outputs=[checkbox, dataset, dataset2, tabs, name, diagnose]
    )
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True, auth=auth_fn)
