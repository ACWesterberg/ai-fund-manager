from fundmgr.config import FeeConfig


def test_fee_minimum():
    fee = FeeConfig()
    assert fee.calc(500) == 1.0   # 0.10% of 500 = 0.50, below min


def test_fee_normal():
    fee = FeeConfig()
    assert fee.calc(2500) == pytest.approx(2.50)
    assert fee.calc(10_000) == pytest.approx(10.00)


def test_fee_maximum():
    fee = FeeConfig()
    assert fee.calc(100_000) == 99.0   # 0.10% of 100k = 100, capped at 99


def test_fee_at_cap_boundary():
    fee = FeeConfig()
    # 99,000 SEK × 0.001 = 99.00 — exactly at cap
    assert fee.calc(99_000) == pytest.approx(99.0)
    # 98,000 SEK × 0.001 = 98.00 — just under cap
    assert fee.calc(98_000) == pytest.approx(98.0)


import pytest
