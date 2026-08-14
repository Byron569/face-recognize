import { useState } from 'react';
import { Modal, Form, Input, Upload, Button, message } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { batchImportFaces } from '../api/faces';

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function FaceRegisterModal({ open, onClose }: Props) {
  const [form] = Form.useForm();
  const [fileList, setFileList] = useState<any[]>([]);
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: (formData: FormData) => batchImportFaces(formData),
    onSuccess: (res) => {
      const results = res.data?.results ?? [];
      const failed = results.filter((r) => r.status === 'error');
      if (results.length > 0 && failed.length === results.length) {
        // 全部失败:不关闭,提示原因
        message.error(`注册失败:${failed[0].reason || '图片质量不达标'}`);
        return;
      }
      if (failed.length > 0) {
        message.warning(
          `${results.length - failed.length} 张成功,${failed.length} 张失败:${failed[0].reason || '质量不达标'}`
        );
      } else {
        message.success(`注册成功(${results.length} 张照片)`);
      }
      queryClient.invalidateQueries({ queryKey: ['faces'] });
      form.resetFields();
      setFileList([]);
      onClose();
    },
    onError: (err: any) => {
      message.error(err.response?.data?.detail || '注册失败');
    },
  });

  const handleOk = () => {
    if (fileList.length === 0) {
      message.error('请至少选择一张照片');
      return;
    }
    form.validateFields().then((values) => {
      const fd = new FormData();
      fd.append('name', values.name);
      fd.append('notes', values.notes || '');
      fileList.forEach((f) => {
        if (f.originFileObj) {
          fd.append('images', f.originFileObj);
        }
      });
      mutation.mutate(fd);
    });
  };

  return (
    <Modal
      title="注册新人脸"
      open={open}
      onOk={handleOk}
      onCancel={onClose}
      confirmLoading={mutation.isPending}
    >
      <Form form={form} layout="vertical">
        <Form.Item name="name" label="姓名" rules={[{ required: true, message: '请输入姓名' }]}>
          <Input placeholder="输入姓名" />
        </Form.Item>
        <Form.Item name="notes" label="备注">
          <Input.TextArea rows={2} />
        </Form.Item>
        <Form.Item label="照片(可多选,建议 3~5 张不同角度)" required>
          <Upload
            listType="picture"
            multiple
            fileList={fileList}
            beforeUpload={(file) => {
              setFileList((prev) => [
                ...prev,
                { uid: file.uid, name: file.name, status: 'done', originFileObj: file },
              ]);
              return false;
            }}
            onRemove={(file) => setFileList((prev) => prev.filter((f) => f.uid !== file.uid))}
          >
            <Button icon={<UploadOutlined />}>选择照片</Button>
          </Upload>
          <div style={{ color: '#999', fontSize: 12, marginTop: 4 }}>
            多角度照片注册后,实时识别更稳;模糊/过小的照片会被自动拒绝
          </div>
        </Form.Item>
      </Form>
    </Modal>
  );
}
