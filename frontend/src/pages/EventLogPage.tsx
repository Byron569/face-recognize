import { useState } from 'react';
import { Table, Tag, Select, Button, Space, message, Card, Badge, Popconfirm } from 'antd';
import { CheckCircleOutlined, WarningOutlined, UserSwitchOutlined, AlertOutlined, ClockCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchEvents, acknowledgeEvent, deleteEvents, deleteAllEvents, EventItem } from '../api/events';
import { eventMeta } from '../config';
import dayjs from 'dayjs';

const eventIconMap: Record<string, React.ReactNode> = {
  warning: <WarningOutlined />,
  alert: <AlertOutlined />,
  check: <CheckCircleOutlined />,
  user: <UserSwitchOutlined />,
  clock: <ClockCircleOutlined />,
};

export default function EventLogPage() {
  const [page, setPage] = useState(1);
  const [typeFilter, setTypeFilter] = useState<string | undefined>();
  const [ackFilter, setAckFilter] = useState<boolean | undefined>(false);
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ['events', page, typeFilter, ackFilter],
    queryFn: () =>
      fetchEvents({
        page,
        page_size: 20,
        event_type: typeFilter,
        acknowledged: ackFilter,
      }),
  });

  const ackMutation = useMutation({
    mutationFn: acknowledgeEvent,
    onSuccess: () => {
      message.success('已确认');
      queryClient.invalidateQueries({ queryKey: ['events'] });
    },
  });

  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([]);

  const deleteMutation = useMutation({
    mutationFn: deleteEvents,
    onSuccess: (res) => {
      message.success(`已删除 ${res.data.deleted} 条记录`);
      setSelectedRowKeys([]);
      queryClient.invalidateQueries({ queryKey: ['events'] });
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '删除失败'),
  });

  const deleteAllMutation = useMutation({
    mutationFn: deleteAllEvents,
    onSuccess: (res) => {
      message.success(`已删除 ${res.data.deleted} 条记录`);
      setSelectedRowKeys([]);
      queryClient.invalidateQueries({ queryKey: ['events'] });
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '删除失败'),
  });

  const handleBatchDelete = () => {
    if (selectedRowKeys.length === 0) return;
    deleteMutation.mutate(selectedRowKeys.map(Number));
  };

  const handleDeleteAll = () => {
    deleteAllMutation.mutate({
      event_type: typeFilter,
      acknowledged: ackFilter,
    });
  };

  const columns = [
    {
      title: '时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: (v: string) => (
        <span style={{ fontSize: 13, color: '#555' }}>
          {dayjs(v).format('YYYY-MM-DD HH:mm:ss')}
        </span>
      ),
    },
    {
      title: '类型',
      dataIndex: 'event_type',
      key: 'event_type',
      width: 130,
      render: (v: string) => {
        const m = eventMeta[v] || { color: '#999', label: v, icon: 'alert' };
        return (
          <Tag color={m.color} icon={eventIconMap[m.icon]} style={{ borderRadius: 4, padding: '2px 8px' }}>
            {m.label}
          </Tag>
        );
      },
    },
    {
      title: '摄像头',
      dataIndex: 'camera_id',
      key: 'camera_id',
      width: 110,
      render: (v: string) => <Tag style={{ borderRadius: 4 }}>{v}</Tag>,
    },
    {
      title: '人员',
      dataIndex: 'identity_name',
      key: 'identity_name',
      width: 100,
      render: (v: string | null) =>
        v ? <strong>{v}</strong> : <span style={{ color: '#bbb' }}>-</span>,
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (v: number) => {
        if (!v || v <= 0) return <span style={{ color: '#bbb' }}>-</span>;
        const pct = (v * 100).toFixed(0);
        const color = Number(pct) >= 80 ? '#52c41a' : Number(pct) >= 60 ? '#fa8c16' : '#ff4d4f';
        return (
          <span style={{ color, fontWeight: 600, fontSize: 14 }}>{pct}%</span>
        );
      },
    },
    {
      title: '状态',
      dataIndex: 'acknowledged',
      key: 'acknowledged',
      width: 110,
      render: (v: boolean) =>
        v ? <Badge status="success" text="已确认" /> : <Badge status="processing" text="未处理" />,
    },
    {
      title: '操作',
      key: 'actions',
      width: 100,
      render: (_: unknown, record: EventItem) =>
        !record.acknowledged && (
          <Button
            type="link"
            size="small"
            icon={<CheckCircleOutlined />}
            onClick={() => ackMutation.mutate(record.id)}
            loading={ackMutation.isPending}
          >
            确认
          </Button>
        ),
    },
  ];

  const items = data?.data.items || [];
  const total = data?.data.total || 0;

  return (
    <Card bordered={false} style={{ borderRadius: 8 }}>
      <Space style={{ marginBottom: 16 }} size="middle">
        <Select
          placeholder="全部类型"
          allowClear
          style={{ width: 150 }}
          value={typeFilter}
          onChange={setTypeFilter}
          options={Object.entries(eventMeta).map(([k, v]) => ({ value: k, label: v.label }))}
        />
        <Select
          placeholder="确认状态"
          style={{ width: 120 }}
          value={ackFilter}
          onChange={setAckFilter}
          options={[
            { value: undefined, label: '全部' },
            { value: false, label: '未处理' },
            { value: true, label: '已确认' },
          ]}
        />
        <Tag color="blue">{total} 条记录</Tag>
        <Popconfirm
          title={`确定删除选中的 ${selectedRowKeys.length} 条记录吗?`}
          description="删除后不可恢复"
          okButtonProps={{ danger: true }}
          onConfirm={handleBatchDelete}
          disabled={selectedRowKeys.length === 0}
        >
          <Button
            danger
            icon={<DeleteOutlined />}
            disabled={selectedRowKeys.length === 0}
            loading={deleteMutation.isPending}
          >
            批量删除{selectedRowKeys.length > 0 ? `(${selectedRowKeys.length})` : ''}
          </Button>
        </Popconfirm>
        <Popconfirm
          title={`确定删除${typeFilter || ackFilter !== undefined ? '当前筛选条件下的' : ''}全部记录吗?`}
          description={`将删除 ${total} 条记录,不可恢复`}
          okButtonProps={{ danger: true }}
          onConfirm={handleDeleteAll}
          disabled={total === 0}
        >
          <Button
            danger
            icon={<DeleteOutlined />}
            disabled={total === 0}
            loading={deleteAllMutation.isPending}
          >
            清空{typeFilter || ackFilter !== undefined ? '筛选结果' : '全部'}
          </Button>
        </Popconfirm>
      </Space>
      <Table
        rowKey="id"
        rowSelection={{
          selectedRowKeys,
          onChange: setSelectedRowKeys,
        }}
        columns={columns}
        dataSource={items}
        loading={isLoading}
        size="middle"
        scroll={{ x: 820 }} // 手机小屏横向滚动,不挤压列宽
        pagination={{
          current: page,
          pageSize: 20,
          total,
          onChange: setPage,
          showSizeChanger: false,
          showTotal: (t) => `共 ${t} 条`,
        }}
        style={{ borderRadius: 8, overflow: 'hidden' }}
      />
    </Card>
  );
}
