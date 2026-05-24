import React from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Svg, { Circle, Line, Path, Polyline, Rect } from 'react-native-svg';
import { colors } from './theme';

export type TabKey = 'device' | 'camera' | 'stream' | 'system' | 'edge' | 'console';

const tabs: { key: TabKey; label: string; icon: (active: boolean) => React.ReactNode }[] = [
  {
    key: 'device',
    label: 'Device',
    icon: (active) => (
      <Svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke={active ? '#7DD89E' : 'rgba(255,255,255,0.45)'} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <Path d="m6.5 6.5 11 11L12 23V1l5.5 5.5-11 11" />
      </Svg>
    ),
  },
  {
    key: 'camera',
    label: 'Camera',
    icon: (active) => (
      <Svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke={active ? '#7DD89E' : 'rgba(255,255,255,0.45)'} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <Path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
        <Circle cx={12} cy={13} r={4} />
      </Svg>
    ),
  },
  {
    key: 'stream',
    label: 'Stream',
    icon: (active) => (
      <Svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke={active ? '#7DD89E' : 'rgba(255,255,255,0.45)'} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <Circle cx={12} cy={12} r={2} />
        <Path d="M16.24 7.76a6 6 0 0 1 0 8.49M7.76 16.24a6 6 0 0 1 0-8.48M20.49 4.93a10 10 0 0 1 0 14.14M3.51 19.07a10 10 0 0 1 0-14.14" />
      </Svg>
    ),
  },
  {
    key: 'system',
    label: 'System',
    icon: (active) => (
      <Svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke={active ? '#7DD89E' : 'rgba(255,255,255,0.45)'} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <Rect x={3} y={3} width={7} height={7} rx={1.5} />
        <Rect x={14} y={3} width={7} height={7} rx={1.5} />
        <Rect x={3} y={14} width={7} height={7} rx={1.5} />
        <Rect x={14} y={14} width={7} height={7} rx={1.5} />
      </Svg>
    ),
  },
  {
    key: 'edge',
    label: 'Edge',
    icon: (active) => (
      <Svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke={active ? '#7DD89E' : 'rgba(255,255,255,0.45)'} strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
        <Path d="M12 2.5 13.6 8 19 9.5 13.6 11 12 16.5 10.4 11 5 9.5 10.4 8 12 2.5z" />
        <Path d="M18.5 14.5 19.2 16.8 21.5 17.5 19.2 18.2 18.5 20.5 17.8 18.2 15.5 17.5 17.8 16.8 18.5 14.5z" />
        <Path d="M5.5 16 6 17.6 7.6 18.1 6 18.6 5.5 20.2 5 18.6 3.4 18.1 5 17.6 5.5 16z" />
      </Svg>
    ),
  },
  {
    key: 'console',
    label: 'Console',
    icon: (active) => (
      <Svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke={active ? '#7DD89E' : 'rgba(255,255,255,0.45)'} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <Polyline points="4 17 10 11 4 5" />
        <Line x1={12} y1={19} x2={20} y2={19} />
      </Svg>
    ),
  },
];

export function TabBar({ active, onChange }: { active: TabKey; onChange: (k: TabKey) => void }) {
  return (
    <View style={styles.outer}>
      {tabs.map((t) => {
        const isActive = t.key === active;
        const inner = (
          <View style={[styles.tab, isActive && styles.tabActive]}>
            {t.icon(isActive)}
            <Text style={[styles.label, { color: isActive ? '#7DD89E' : 'rgba(255,255,255,0.45)', fontWeight: isActive ? '700' : '500' }]}>{t.label}</Text>
          </View>
        );
        return (
          <Pressable key={t.key} onPress={() => onChange(t.key)} style={styles.tabWrap}>
            {inner}
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  outer: {
    position: 'absolute',
    left: 12,
    right: 12,
    bottom: 12,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(15,20,20,0.72)',
    borderRadius: 30,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    padding: 8,
    gap: 4,
    shadowColor: '#000',
    shadowOpacity: 0.4,
    shadowOffset: { width: 0, height: 14 },
    shadowRadius: 44,
    elevation: 12,
  },
  tabWrap: { flex: 1 },
  tab: {
    paddingVertical: 10,
    paddingHorizontal: 8,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    borderRadius: 22,
  },
  tabActive: {
    backgroundColor: 'rgba(125,216,158,0.18)',
  },
  gradientTab: {
    paddingVertical: 10,
    paddingHorizontal: 8,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    borderRadius: 22,
  },
  label: {
    fontSize: 10,
    lineHeight: 12,
  },
});
