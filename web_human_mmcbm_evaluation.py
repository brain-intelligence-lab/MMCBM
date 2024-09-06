# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com

0. Concept （已做，独立组）


(no)                1. Concept + Label  (5)
(label)             2. Concept + Label random (5)
(concept)           3. Concept random + Label (5)
(concept+label)     4. Concept random + Label random (5)
"""

import os.path

import gradio as gr
import pandas as pd
import torch

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

max_k = 20
github = """
               <div style="display: flex; align-items: center; justify-content: center; height: 100px;">
                    <a href='https://github.com/ly1998117/MMCBM' target='_blank' style="text-decoration: none; color: black; text-align: center;">
                        <img src='https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png' width='50' height='50' style="display: block; margin: 0 auto;"/>
                        GitHub repository
                    </a>
                </div>
           """
texts = {
    'title': {'en': '# MMCBM_2 Interface', 'cn': '# MMCBM界面'},
    'desc': {
        'en': "### Prediction: Upload Fundus Images, Click Predict button to get the Top-10 concepts and prediction. \n"
              "### Intervention: After adjusting the sliders, "
              "click the 'Intervention' button to update the prediction.",
        'cn': '### 预测：上传眼单击一行跳转模型预测，获取前10个概念和预测。\n'
              '### 评估：参考模型预测，给出你自己的预测。 \n'
              '### 干预：调整滑块后，单击“干预”按钮可以更新预测。\n'
              '注意：根据实验需求，部分病人的模型预测结果为随机值！'},
    'predict': {'en': 'Predict', 'cn': '预测'},
    'intervene': {'en': 'Intervene Concept', 'cn': '干预概念'},
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
btn_intervene = gr.Button(value="概念干预")


def update_texts(language):
    return [gr.Markdown(texts['title'][language]),
            gr.Markdown(texts['desc'][language]),
            gr.Button(value=texts['intervene'][language])
            ]


class Session:
    def __init__(self, value, fn=lambda x: x):
        self.state = gr.State(value=value)
        self.fn = fn

    def get_user_name(self):
        request = gr.Request()
        return request.username

    @property
    def value(self):
        return self.state.value

    def __call__(self):
        return self.state


class EasyDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class UserSession:
    def __init__(self, data, users, default='root'):
        self.root = 'CSV/human_evaluation/MMCBM_2'
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

        def _divide(x):
            x['user'] = self.table_cut(length=len(x), labels=self.users)
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
                    '_load_data', 'table_cut', 'divide_by_users', 'state']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

        # 尝试从 self.state[user.value] 返回属性值
        try:
            return self.state[self.current_user][item]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object User {self.current_user} has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key in ['root', 'default', 'current_user', 'data', 'users', 'state',
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
    json_path='result/CAV_m2CBM_sigmoid_C0.1CrossEntropy_32_report_strict_aow_zero_MM_max/fold_0_report_strict_r1.0_c1.0',
    backbone='Efficientb0_SCLS_attnscls_CrossEntropy_32/fold_0',
    idx=180,
    bidx=180,
    device='cpu',
    normalize='linear',
)
session = UserSession(data=predict.get_test_data(mask=False, format='dataframe'), users=list(user_passwd.keys()))
session.top_k = 10
session.bottom_k = 10
session.language = "cn"
session.attn = torch.empty((1, 3, 103))
callback = gr.CSVLogger()


########################### Functions ###########################
def is_concept_random(rand):
    return 'concept' in rand


def is_label_random(rand):
    return 'label' in rand


def is_any_random(rand):
    return is_concept_random(rand) or is_label_random(rand)


def is_all_random(rand):
    return is_concept_random(rand) and is_label_random(rand)


def auth_fn(user, passwd):
    if user not in user_passwd.keys():
        return False
    return user_passwd[user] == passwd


def load_fn(request: gr.Request):
    return request.username


def load_param(user):
    return session[user].top_k, session[user].bottom_k, session[user].language, gr.Dataset(
        samples=session[user].get_data()), gr.Dataset(
        samples=session[user].get_data(diagnosed=True))


def dataset_fn(user, data_id):
    data = session[user].get_data()[data_id]
    return data[1], data[2], *data[5:], session[user].is_random_from_id(data_id)


def dataset2_fn(user, data_id):
    data = session[user].get_data(diagnosed=True)[data_id]
    return data[1], data[2], *data[5:], session[user].is_random_from_id(data_id, diagnosed=True)


def top_k_drop_fn(user, x):
    session[user].top_k = x


def bottom_k_drop_fn(user, x):
    session[user].bottom_k = x


def language_fn(user, x):
    session[user].language = x


def predict_topk_concept(user, is_random, *args):
    args += (session[user].top_k, session[user].language,)
    predict.set_random(is_concept_random(is_random))
    return predict.predict_topk_concept(*args)


def predict_bottomk_concept(user, is_random):
    predict.set_random(is_concept_random(is_random))
    return predict.predict_bottomk_concept(session[user].bottom_k)


def predict_label(user, is_random, name, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us):
    if predict.get_random() and not is_label_random(is_random):
        predict.set_random(is_label_random(is_random))
        predict.predict_topk_concept(name, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us,
                                     session[user].top_k, session[user].language, )
    predict.set_random(is_label_random(is_random))
    return predict.predict_label(session[user].language)


def get_attention_matrix(user, is_random):
    predict.random = is_concept_random(is_random)
    session[user].attn = predict.get_attention_matrix()


def set_attention_matrix(user, is_random):
    predict.random = is_concept_random(is_random)
    predict.set_attention_matrix(session[user].attn)


def modify(user, is_random, *args):
    predict.set_random(is_concept_random(is_random))
    args += (session[user].top_k, session[user].bottom_k, session[user].language)
    return predict.modify(*args)


def checkbox_fn(x):
    return gr.Text(value=x)


def diagnose_fn(user, diagnose, name, top_k_drop, bottom_k_drop, is_random, label):
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
    callback.flag([
        name, top_k_drop, bottom_k_drop, is_random, label
    ], username=user, flag_option=diagnose)
    return (gr.Radio(value=None), gr.Dataset(samples=session[user].get_data()),
            gr.Dataset(samples=session[user].get_data(diagnosed=True)), gr.Tabs(selected=0), '', '待诊断')


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
            top_k_drop = gr.Dropdown(value=session.top_k, label="前 K 概念",
                                     choices=[i for i in range(5, max_k + 1, 5)],
                                     multiselect=False,
                                     min_width=1)
        with gr.Column(scale=1, min_width=1):
            bottom_k_drop = gr.Dropdown(value=session.bottom_k, label="后 K 概念",
                                        choices=[i for i in range(5, max_k + 1, 5)], multiselect=False,
                                        min_width=1)
        with gr.Column(scale=1, min_width=1):
            lan = gr.Dropdown(label="语言", value=session.language,
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
                with gr.Column(scale=2, min_width=160):
                    with gr.Accordion("前 K 个概念", open=True):
                        sliders = [gr.Slider(step=0.01, label=None) if i < session.top_k
                                   else gr.Slider(step=0.01, label=None, visible=False) for i in range(max_k)]
                with gr.Column(scale=2, min_width=160):
                    with gr.Accordion("后 K 个概念", open=True):
                        bottom_sliders = [gr.Slider(step=0.01, label=None) if i < session.bottom_k
                                          else gr.Slider(step=0.01, label=None, visible=False) for i in range(max_k)]

                with gr.Column(scale=3, min_width=160):
                    with gr.Accordion("模型预测", open=True, elem_id="output-panel"):
                        with gr.Row():
                            label = gr.Label(num_top_classes=3)
                        with gr.Row():
                            btn_intervene.render()

                    with gr.Accordion("人工诊断", open=True, elem_id="human-panel"):
                        with gr.Row():
                            checkbox = gr.Radio(label='选择疾病，确保只选中一个疾病',
                                                choices=['血管瘤', '转移癌', '黑色素瘤'])
                        with gr.Row():
                            diagnose_btn = gr.Button(value="确定")
    ############################################## Trigger ##############################################
    demo.load(load_fn, inputs=None, outputs=user).then(load_param, inputs=user,
                                                       outputs=[top_k_drop, bottom_k_drop, lan, dataset, dataset2])
    callback.setup([name, top_k_drop, bottom_k_drop, is_random, label],
                   flagging_dir=f"{session.root}/flagged_mmcbm_data_points")
    top_k_drop.change(fn=top_k_drop_fn, inputs=[user, top_k_drop])
    bottom_k_drop.change(fn=bottom_k_drop_fn, inputs=[user, bottom_k_drop])
    lan.change(update_texts, inputs=lan,
               outputs=[title, desc, btn_intervene]).then(
        fn=language_fn, inputs=[user, lan]
    )
    predict.set_topk_sliders(sliders)
    predict.set_bottomk_sliders(bottom_sliders)

    dataset.click(
        fn=dataset_fn,
        inputs=[user, dataset],
        outputs=[name, diagnose, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us, is_random]
    ).then(
        fn=lambda x: gr.Tabs(selected=1),
        inputs=None,
        outputs=tabs,
    ).then(fn=predict_topk_concept,
           inputs=[user, is_random, name, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us],
           outputs=sliders).then(
        fn=predict_bottomk_concept,
        inputs=[user, is_random],
        outputs=bottom_sliders).then(
        fn=predict_label,
        inputs=[user, is_random, name, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us],
        outputs=label).then(
        fn=get_attention_matrix,
        inputs=[user, is_random],
    )

    dataset2.click(
        fn=dataset2_fn,
        inputs=[user, dataset2],
        outputs=[name, diagnose, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us, is_random]
    ).then(
        fn=lambda x: gr.Tabs(selected=1),
        inputs=None,
        outputs=tabs,
    ).then(fn=predict_topk_concept,
           inputs=[user, is_random, name, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us],
           outputs=sliders).then(
        fn=predict_bottomk_concept,
        inputs=[user, is_random],
        outputs=bottom_sliders).then(
        fn=predict_label,
        inputs=[user, is_random, name, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us],
        outputs=label).then(
        fn=get_attention_matrix,
        inputs=[user, is_random],
    )

    btn_intervene.click(
        fn=set_attention_matrix,
        inputs=[user, is_random],
    ).then(
        fn=modify,
        inputs=[user, is_random] + sliders + bottom_sliders,
        outputs=label
    )

    checkbox.change(
        fn=checkbox_fn,
        inputs=checkbox,
        outputs=diagnose
    )
    diagnose_btn.click(
        fn=diagnose_fn,
        inputs=[user, diagnose, name, top_k_drop, bottom_k_drop, is_random, label],
        outputs=[checkbox, dataset, dataset2, tabs, name, diagnose]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7861, share=True, auth=auth_fn)
