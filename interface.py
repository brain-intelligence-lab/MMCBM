import gradio as gr
from web.intervention import Intervention

patients = ['马平社', '谢春艳', '邵佳南', '周庆玉', '何德翠', '宋志忠D', '祝丽华D', '王湛铭', '李启明', '武元春D',
            '王振铎', '叶爱芳', '洪文远', '王胖子D', '汪育元', '郑桂荣', '王海', '韩志袁', '关金平']
top_k = 10
bottom_k = 10
max_k = 20
texts = [
    {'en': '# Intervention Interface', 'cn': '# 干预界面'},
    {'en': "### Prediction: Upload Fundus Images, Click Predict button to get the Top-10 concepts and prediction. \n"
           "### Intervention: After adjusting the sliders, "
           "click the 'Intervention' button to update the prediction.",
     'cn': '### 预测：上传眼底图像，单击预测按钮以获取前10个概念和预测。\n'
           '### 干预：调整滑块后，单击“干预”按钮以更新预测。'},

    {'en': 'Predict', 'cn': '预测'},
    {'en': 'Intervene Concept', 'cn': '干预概念'},
    {'en': 'Generate Report', 'cn': '生成报告'},
]

name = gr.Text(label='Name', min_width=2, scale=1)
pathology = gr.Text(label='Pathology', min_width=2, scale=1)
# fa_e_label = gr.HTML(
#     value="<div style='position: relative;top: -1px;'>"
#           "<span style='position: absolute; left: 0; top: 0; padding: 0 10px; font-size: 14px; color: #6b727f; z-index:10;"
#           "border-radius: 5px;border: 1px solid #e6e6e6; background-color:white'>FA-E</span>"
#           "</div>")
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
fa_e = gr.Image(type="pil", label='FA-E', min_width=2, scale=1, height=120, show_label=False)
fa_m = gr.Image(type="pil", label='FA-M', min_width=2, scale=1, height=120, show_label=False)
fa_l = gr.Image(type="pil", label='FA-L', min_width=2, scale=1, height=120, show_label=False)
icga_e = gr.Image(type="pil", label='ICGA-E', min_width=2, scale=1, height=120, show_label=False)
icga_m = gr.Image(type="pil", label='ICGA-M', min_width=2, scale=1, height=120, show_label=False)
icga_l = gr.Image(type="pil", label='ICGA-L', min_width=2, scale=1, height=120, show_label=False)
us = gr.Image(type="pil", label='US', min_width=2, show_label=False, show_download_button=False)
# buttons
btn_predict = gr.Button(value="Predict")
btn_intervene = gr.Button(value="Intervene Concept")
btn_report = gr.Button(value="Generate Report")

predict = Intervention(
    json_path='result/CAV_m2CBM_sigmoid_C0.1CrossEntropy_32_report_strict_add_aow_zero_MM_max/fold_0_report_strict_r1.0_c1.0',
    backbone='Efficientb0_SCLS_attnscls_CrossEntropy_32/fold_0',
    idx=180,
    device='cpu',
    normalize='default',
    top_k=top_k,
    bottom_k=bottom_k,
)


def update_texts(language):
    predict.set_language(language)
    return [gr.Markdown(texts[0][predict.language]),
            gr.Markdown(texts[1][predict.language]),
            gr.Button(value=texts[2][predict.language]),
            gr.Button(value=texts[3][predict.language]),
            gr.Button(value=texts[4][predict.language])]


with gr.Blocks() as demo:
    title = gr.Markdown(texts[0][predict.language])
    with gr.Row():
        with gr.Column(scale=50):
            desc = gr.Markdown(texts[1][predict.language])
        with gr.Column(scale=1):
            top_k = gr.Dropdown(value=top_k, label="Top-K Concepts",
                                choices=[i for i in range(5, max_k + 1, 5)], multiselect=False)
            top_k.change(predict.change_top_k, inputs=[top_k])
        with gr.Column(scale=1):
            bottom_k = gr.Dropdown(value=bottom_k, label="Bottom-K Concepts",
                                   choices=[i for i in range(5, max_k + 1, 5)], multiselect=False)
            bottom_k.change(predict.change_bottom_k, inputs=[bottom_k])
        with gr.Column(scale=1):
            lan = gr.Dropdown(label="Language", value='en', choices=["en", "cn"], elem_id="language",
                              multiselect=False)
            lan.change(update_texts, inputs=[lan], outputs=[title, desc, btn_predict, btn_intervene, btn_report])

    with gr.Row():
        with gr.Accordion("Image Examples, Click to apply", open=True, elem_id="input-panel"):
            gr.Examples(
                examples=predict.get_test_data(num_of_each_pathology=1, mask=False,
                                               names=patients),
                inputs=[name, pathology, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us],  # type: ignore
                outputs=None,  # type: ignore
                label=None,
                examples_per_page=4,
            )

    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            with gr.Accordion("Different modal images, please click to upload.", open=True, elem_id="input-panel"):
                with gr.Row(equal_height=True):
                    name.render()
                    pathology.render()
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
                with gr.Row():
                    gr.ClearButton([fa_e, fa_m, fa_l], value="Clear FA", min_width=1)
                    gr.ClearButton([icga_e, icga_m, icga_l], value="Clear ICGA", min_width=1)
                    gr.ClearButton([us], value="Clear US", min_width=1)
                with gr.Row():
                    gr.ClearButton([fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us], value="Clear All")
        with gr.Column(scale=2):
            with gr.Accordion("Top-K", open=True):
                sliders = [gr.Slider(step=0.01, label=None) if i < predict.top_k
                           else gr.Slider(step=0.01, label=None, visible=False) for i in range(max_k)]
        with gr.Column(scale=2):
            with gr.Accordion("Bottom-K", open=True):
                bottom_sliders = [gr.Slider(step=0.01, label=None) if i < predict.bottom_k
                                  else gr.Slider(step=0.01, label=None, visible=False) for i in range(max_k)]

        with gr.Column(scale=4):
            with gr.Accordion("Output", open=True, elem_id="output-panel"):
                with gr.Row():
                    label = gr.Label(num_top_classes=3)
                with gr.Row():
                    chatbot = gr.Chatbot(label=f"当前模型：ChatGPT-3.5", elem_id="gpt-chatbot", layout='panel')
                with gr.Row():
                    download = gr.File(label="Download")
                with gr.Row():
                    btn_predict.render()
                    btn_intervene.render()
                    btn_report.render()
                    clear = gr.ClearButton([chatbot, *sliders, *bottom_sliders, label, download])
    with gr.Row():
        plot = gr.BarPlot(show_label=False)
        clear.add(plot)

    predict.set_topk_sliders(sliders)
    predict.set_bottomk_sliders(bottom_sliders)
    btn_predict.click(fn=predict.predict_topk_concept,
                      inputs=[name, pathology, fa_e, fa_m, fa_l, icga_e, icga_m, icga_l, us],
                      outputs=sliders).then(fn=predict.predict_bottomk_concept,
                                            outputs=bottom_sliders).then(fn=predict.predict_label, inputs=None,
                                                                         outputs=label).then(
        fn=predict.fresh_barplot,
        inputs=None,
        outputs=plot).then(fn=predict.download('Intervention-concepts.csv'), outputs=download)
    btn_intervene.click(fn=predict.modify, inputs=sliders + bottom_sliders, outputs=label).then(
        fn=predict.download('Intervention-concepts-modify.csv'), outputs=download)
    btn_report.click(fn=predict.report, inputs=chatbot, outputs=chatbot)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
