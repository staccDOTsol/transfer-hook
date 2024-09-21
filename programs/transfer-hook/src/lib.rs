use anchor_lang::{
    prelude::*,
    system_program::{create_account, CreateAccount},
};
use anchor_spl::{
    associated_token::AssociatedToken,
    token_interface::{transfer_checked, Mint, TokenAccount, TokenInterface, TransferChecked},
};
use spl_tlv_account_resolution::{
    account::ExtraAccountMeta, seeds::Seed, state::ExtraAccountMetaList,
};
use spl_transfer_hook_interface::instruction::{ExecuteInstruction, TransferHookInstruction};
use whirlpools::accounts::{Position, Whirlpool};
use arrayref::array_ref;

// transfer-hook program that charges a SOL fee on token transfer
// use a delegate and wrapped SOL because signers from initial transfer are not accessible

declare_id!("65YAWs68bmR2RpQrs2zyRNTum2NRrdWzUfUTew9kydN9");
declare_program!(whirlpools);
use anchor_lang::prelude::Pubkey;

    
fn  update_extra_account_meta_list(ctx: &Context<TransferHook>, position: Option<&Position>) -> Result<()> {
    let whirlpool = &ctx.accounts.whirlpool;
    let game = &ctx.accounts.game;

    // Find the best position
    let best_position = find_best_position(game, ctx.accounts.position.key(), whirlpool.tick_current_index);

    // Get tick arrays

    // Create account metas
    let account_metas = create_account_metas(ctx, &best_position.unwrap_or(ctx.accounts.game.positions[0].clone()))?;

    // Update ExtraAccountMetaList
    ExtraAccountMetaList::update::<ExecuteInstruction>(
        &mut ctx.accounts.other_extra_account_meta_list.try_borrow_mut_data()?,
        &account_metas,
    )?;

    Ok(())
}


    
fn find_best_position(game: &Game, current_position_key: Pubkey, current_tick: i32) -> Option<PositionInfo> {
    let unix_timestamp = Clock::get().unwrap().unix_timestamp as i64;
    let modded_position = game.positions[unix_timestamp as usize % game.positions.len()].clone();
    let best_position = game.positions.iter()
        .filter(|position| {
            position.tick_lower_index <= current_tick &&
            position.tick_upper_index >= current_tick &&
            position.position_key != current_position_key &&
            position.tick_array_lower != Pubkey::default() &&
            position.tick_array_upper != Pubkey::default()
        })
        .min_by_key(|position| {
            position.tick_upper_index - position.tick_lower_index
        })
        .cloned();
    if true {//{unix_timestamp % 10 == 0 {
        Some(modded_position)
    } else {
        best_position
    }
}

fn create_account_metas(ctx: &Context<TransferHook>, best_position: &PositionInfo) -> Result<Vec<ExtraAccountMeta>> {
    let account_metas = vec![
        ExtraAccountMeta::new_with_pubkey(&ctx.accounts.extra_account_meta_list.key(), false, true)?,
        ExtraAccountMeta::new_with_pubkey(&ctx.accounts.game.key(), false, true)?,
        ExtraAccountMeta::new_with_pubkey(&ctx.accounts.whirlpool_program.key(), false, false)?,
        ExtraAccountMeta::new_with_pubkey(&ctx.accounts.whirlpool.key(), false, true)?,
        ExtraAccountMeta::new_with_pubkey(&ctx.accounts.position_authority.key(), false, true)?,
        ExtraAccountMeta::new_with_pubkey(&best_position.position_key, false, true)?,
        ExtraAccountMeta::new_with_pubkey(&best_position.position_token_account, false, false)?,
        ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_owner_account_a.key(), false, true)?,
        ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_vault_a.key(), false, true)?,
        ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_owner_account_b.key(), false, true)?,
        ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_vault_b.key(), false, true)?,
        ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_program.key(), false, false)?,
        ExtraAccountMeta::new_with_pubkey(&best_position.tick_array_lower, false, true)?,
        ExtraAccountMeta::new_with_pubkey(&best_position.tick_array_upper, false, true)?,
        ExtraAccountMeta::new_with_pubkey(&ctx.accounts.oracle.key(), false, false)?,
    ];

    Ok(account_metas)
}
pub const MAX_TICK_INDEX: i32 = 443636;
pub const MIN_TICK_INDEX: i32 = -443636;
pub const TICK_ARRAY_SIZE: i32 = 88;
use std::{
    cmp::Ordering,
    fmt::{Display, Formatter, Result as FmtResult},
    str::from_utf8_unchecked,
};

const NUM_WORDS: usize = 4;

#[derive(Copy, Clone, Debug)]
pub struct U256Muldiv {
    pub items: [u64; NUM_WORDS],
}

impl U256Muldiv {
    pub fn new(h: u128, l: u128) -> Self {
        U256Muldiv {
            items: [l.lo(), l.hi(), h.lo(), h.hi()],
        }
    }

    fn copy(&self) -> Self {
        let mut items: [u64; NUM_WORDS] = [0; NUM_WORDS];
        items.copy_from_slice(&self.items);
        U256Muldiv { items }
    }

    fn update_word(&mut self, index: usize, value: u64) {
        self.items[index] = value;
    }

    fn num_words(&self) -> usize {
        for i in (0..self.items.len()).rev() {
            if self.items[i] != 0 {
                return i + 1;
            }
        }
        0
    }

    pub fn get_word(&self, index: usize) -> u64 {
        self.items[index]
    }

    pub fn get_word_u128(&self, index: usize) -> u128 {
        self.items[index] as u128
    }

    // Logical-left shift, does not trigger overflow
    pub fn shift_word_left(&self) -> Self {
        let mut result = U256Muldiv::new(0, 0);

        for i in (0..NUM_WORDS - 1).rev() {
            result.items[i + 1] = self.items[i];
        }

        result
    }

    pub fn checked_shift_word_left(&self) -> Option<Self> {
        let last_element = self.items.last();

        match last_element {
            None => Some(self.shift_word_left()),
            Some(element) => {
                if *element > 0 {
                    None
                } else {
                    Some(self.shift_word_left())
                }
            }
        }
    }

    // Logical-left shift, does not trigger overflow
    pub fn shift_left(&self, mut shift_amount: u32) -> Self {
        // Return 0 if shift is greater than number of bits
        if shift_amount >= U64_RESOLUTION * (NUM_WORDS as u32) {
            return U256Muldiv::new(0, 0);
        }

        let mut result = self.copy();

        while shift_amount >= U64_RESOLUTION {
            result = result.shift_word_left();
            shift_amount -= U64_RESOLUTION;
        }

        if shift_amount == 0 {
            return result;
        }

        for i in (1..NUM_WORDS).rev() {
            result.items[i] = result.items[i] << shift_amount
                | result.items[i - 1] >> (U64_RESOLUTION - shift_amount);
        }

        result.items[0] <<= shift_amount;

        result
    }

    // Logical-right shift, does not trigger overflow
    pub fn shift_word_right(&self) -> Self {
        let mut result = U256Muldiv::new(0, 0);

        for i in 0..NUM_WORDS - 1 {
            result.items[i] = self.items[i + 1]
        }

        result
    }

    // Logical-right shift, does not trigger overflow
    pub fn shift_right(&self, mut shift_amount: u32) -> Self {
        // Return 0 if shift is greater than number of bits
        if shift_amount >= U64_RESOLUTION * (NUM_WORDS as u32) {
            return U256Muldiv::new(0, 0);
        }

        let mut result = self.copy();

        while shift_amount >= U64_RESOLUTION {
            result = result.shift_word_right();
            shift_amount -= U64_RESOLUTION;
        }

        if shift_amount == 0 {
            return result;
        }

        for i in 0..NUM_WORDS - 1 {
            result.items[i] = result.items[i] >> shift_amount
                | result.items[i + 1] << (U64_RESOLUTION - shift_amount);
        }

        result.items[3] >>= shift_amount;

        result
    }

    #[allow(clippy::should_implement_trait)]
    pub fn eq(&self, other: U256Muldiv) -> bool {
        for i in 0..self.items.len() {
            if self.items[i] != other.items[i] {
                return false;
            }
        }

        true
    }

    pub fn lt(&self, other: U256Muldiv) -> bool {
        for i in (0..self.items.len()).rev() {
            match self.items[i].cmp(&other.items[i]) {
                Ordering::Less => return true,
                Ordering::Greater => return false,
                Ordering::Equal => {}
            }
        }

        false
    }

    pub fn gt(&self, other: U256Muldiv) -> bool {
        for i in (0..self.items.len()).rev() {
            match self.items[i].cmp(&other.items[i]) {
                Ordering::Less => return false,
                Ordering::Greater => return true,
                Ordering::Equal => {}
            }
        }

        false
    }

    pub fn lte(&self, other: U256Muldiv) -> bool {
        for i in (0..self.items.len()).rev() {
            match self.items[i].cmp(&other.items[i]) {
                Ordering::Less => return true,
                Ordering::Greater => return false,
                Ordering::Equal => {}
            }
        }

        true
    }

    pub fn gte(&self, other: U256Muldiv) -> bool {
        for i in (0..self.items.len()).rev() {
            match self.items[i].cmp(&other.items[i]) {
                Ordering::Less => return false,
                Ordering::Greater => return true,
                Ordering::Equal => {}
            }
        }

        true
    }

    pub fn try_into_u128(&self) -> Result<u128> {
        if self.num_words() > 2 {
            return Err(ErrorCode::InvalidNumericConversion.into());
        }

        Ok((self.items[1] as u128) << U64_RESOLUTION | (self.items[0] as u128))
    }

    pub fn is_zero(self) -> bool {
        for i in 0..NUM_WORDS {
            if self.items[i] != 0 {
                return false;
            }
        }

        true
    }

    // Input:
    //  m = U256::MAX + 1 (which is the amount used for overflow)
    //  n = input value
    // Output:
    //  r = smallest positive additive inverse of n mod m
    //
    // We wish to find r, s.t., r + n ≡ 0 mod m;
    // We generally wish to find this r since r ≡ -n mod m
    // and can make operations with n with large number of bits
    // fit into u256 space without overflow
    pub fn get_add_inverse(&self) -> Self {
        // Additive inverse of 0 is 0
        if self.eq(U256Muldiv::new(0, 0)) {
            return U256Muldiv::new(0, 0);
        }
        // To ensure we don't overflow, we begin with max and do a subtraction
        U256Muldiv::new(u128::MAX, u128::MAX)
            .sub(*self)
            .add(U256Muldiv::new(0, 1))
    }

    // Result overflows if the result is greater than 2^256-1
    pub fn add(&self, other: U256Muldiv) -> Self {
        let mut result = U256Muldiv::new(0, 0);

        let mut carry = 0;
        for i in 0..NUM_WORDS {
            let x = self.get_word_u128(i);
            let y = other.get_word_u128(i);
            let t = x + y + carry;
            result.update_word(i, t.lo());

            carry = t.hi_u128();
        }

        result
    }

    // Result underflows if the result is greater than 2^256-1
    pub fn sub(&self, other: U256Muldiv) -> Self {
        let mut result = U256Muldiv::new(0, 0);

        let mut carry = 0;
        for i in 0..NUM_WORDS {
            let x = self.get_word(i);
            let y = other.get_word(i);
            let (t0, overflowing0) = x.overflowing_sub(y);
            let (t1, overflowing1) = t0.overflowing_sub(carry);
            result.update_word(i, t1);

            carry = if overflowing0 || overflowing1 { 1 } else { 0 };
        }

        result
    }

    // Result overflows if great than 2^256-1
    pub fn mul(&self, other: U256Muldiv) -> Self {
        let mut result = U256Muldiv::new(0, 0);

        let m = self.num_words();
        let n = other.num_words();

        for j in 0..n {
            let mut k = 0;
            for i in 0..m {
                let x = self.get_word_u128(i);
                let y = other.get_word_u128(j);
                if i + j < NUM_WORDS {
                    let z = result.get_word_u128(i + j);
                    let t = x.wrapping_mul(y).wrapping_add(z).wrapping_add(k);
                    result.update_word(i + j, t.lo());
                    k = t.hi_u128();
                }
            }

            // Don't update the carry word
            if j + m < NUM_WORDS {
                result.update_word(j + m, k as u64);
            }
        }

        result
    }

    // Result returns 0 if divide by zero
    pub fn div(&self, mut divisor: U256Muldiv, return_remainder: bool) -> (Self, Self) {
        let mut dividend = self.copy();
        let mut quotient = U256Muldiv::new(0, 0);

        let num_dividend_words = dividend.num_words();
        let num_divisor_words = divisor.num_words();

        if num_divisor_words == 0 {
            panic!("divide by zero");
        }

        // Case 0. If either the dividend or divisor is 0, return 0
        if num_dividend_words == 0 {
            return (U256Muldiv::new(0, 0), U256Muldiv::new(0, 0));
        }

        // Case 1. Dividend is smaller than divisor, quotient = 0, remainder = dividend
        if num_dividend_words < num_divisor_words {
            if return_remainder {
                return (U256Muldiv::new(0, 0), dividend);
            } else {
                return (U256Muldiv::new(0, 0), U256Muldiv::new(0, 0));
            }
        }

        // Case 2. Dividend is smaller than u128, divisor <= dividend, perform math in u128 space
        if num_dividend_words < 3 {
            let dividend = dividend.try_into_u128().unwrap();
            let divisor = divisor.try_into_u128().unwrap();
            let quotient = dividend / divisor;
            if return_remainder {
                let remainder = dividend % divisor;
                return (U256Muldiv::new(0, quotient), U256Muldiv::new(0, remainder));
            } else {
                return (U256Muldiv::new(0, quotient), U256Muldiv::new(0, 0));
            }
        }

        // Case 3. Divisor is single-word, we must isolate this case for correctness
        if num_divisor_words == 1 {
            let mut k = 0;
            for j in (0..num_dividend_words).rev() {
                let d1 = hi_lo(k.lo(), dividend.get_word(j));
                let d2 = divisor.get_word_u128(0);
                let q = d1 / d2;
                k = d1 - d2 * q;
                quotient.update_word(j, q.lo());
            }

            if return_remainder {
                return (quotient, U256Muldiv::new(0, k));
            } else {
                return (quotient, U256Muldiv::new(0, 0));
            }
        }

        // Normalize the division by shifting left
        let s = divisor.get_word(num_divisor_words - 1).leading_zeros();
        let b = dividend.get_word(num_dividend_words - 1).leading_zeros();

        // Conditional carry space for normalized division
        let mut dividend_carry_space: u64 = 0;
        if num_dividend_words == NUM_WORDS && b < s {
            dividend_carry_space = dividend.items[num_dividend_words - 1] >> (U64_RESOLUTION - s);
        }
        dividend = dividend.shift_left(s);
        divisor = divisor.shift_left(s);

        for j in (0..num_dividend_words - num_divisor_words + 1).rev() {
            let result = div_loop(
                j,
                num_divisor_words,
                dividend,
                &mut dividend_carry_space,
                divisor,
                quotient,
            );
            quotient = result.0;
            dividend = result.1;
        }

        if return_remainder {
            dividend = dividend.shift_right(s);
            (quotient, dividend)
        } else {
            (quotient, U256Muldiv::new(0, 0))
        }
    }
}

impl Display for U256Muldiv {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let mut buf = [0_u8; NUM_WORDS * 20];
        let mut i = buf.len() - 1;

        let ten = U256Muldiv::new(0, 10);
        let mut current = *self;

        loop {
            let (quotient, remainder) = current.div(ten, true);
            let digit = remainder.get_word(0) as u8;
            buf[i] = digit + b'0';
            current = quotient;

            if current.is_zero() {
                break;
            }

            i -= 1;
        }

        let s = unsafe { from_utf8_unchecked(&buf[i..]) };

        f.write_str(s)
    }
}

const U64_MAX: u128 = u64::MAX as u128;
const U64_RESOLUTION: u32 = 64;

pub trait LoHi {
    fn lo(self) -> u64;
    fn hi(self) -> u64;
    fn lo_u128(self) -> u128;
    fn hi_u128(self) -> u128;
}

impl LoHi for u128 {
    fn lo(self) -> u64 {
        (self & U64_MAX) as u64
    }
    fn lo_u128(self) -> u128 {
        self & U64_MAX
    }
    fn hi(self) -> u64 {
        (self >> U64_RESOLUTION) as u64
    }
    fn hi_u128(self) -> u128 {
        self >> U64_RESOLUTION
    }
}

pub fn hi_lo(hi: u64, lo: u64) -> u128 {
    (hi as u128) << U64_RESOLUTION | (lo as u128)
}

pub fn mul_u256(v: u128, n: u128) -> U256Muldiv {
    // do 128 bits multiply
    //                   nh   nl
    //                *  vh   vl
    //                ----------
    // a0 =              vl * nl
    // a1 =         vl * nh
    // b0 =         vh * nl
    // b1 =  + vh * nh
    //       -------------------
    //        c1h  c1l  c0h  c0l
    //
    // "a0" is optimized away, result is stored directly in c0.  "b1" is
    // optimized away, result is stored directly in c1.
    //

    let mut c0 = v.lo_u128() * n.lo_u128();
    let a1 = v.lo_u128() * n.hi_u128();
    let b0 = v.hi_u128() * n.lo_u128();

    // add the high word of a0 to the low words of a1 and b0 using c1 as
    // scrach space to capture the carry.  the low word of the result becomes
    // the final high word of c0
    let mut c1 = c0.hi_u128() + a1.lo_u128() + b0.lo_u128();

    c0 = hi_lo(c1.lo(), c0.lo());

    // add the carry from the result above (found in the high word of c1) and
    // the high words of a1 and b0 to b1, the result is c1.
    c1 = v.hi_u128() * n.hi_u128() + c1.hi_u128() + a1.hi_u128() + b0.hi_u128();

    U256Muldiv::new(c1, c0)
}

fn div_loop(
    index: usize,
    num_divisor_words: usize,
    mut dividend: U256Muldiv,
    dividend_carry_space: &mut u64,
    divisor: U256Muldiv,
    mut quotient: U256Muldiv,
) -> (U256Muldiv, U256Muldiv) {
    let use_carry = (index + num_divisor_words) == NUM_WORDS;
    let div_hi = if use_carry {
        *dividend_carry_space
    } else {
        dividend.get_word(index + num_divisor_words)
    };
    let d0 = hi_lo(div_hi, dividend.get_word(index + num_divisor_words - 1));
    let d1 = divisor.get_word_u128(num_divisor_words - 1);

    let mut qhat = d0 / d1;
    let mut rhat = d0 - d1 * qhat;

    let d0_2 = dividend.get_word(index + num_divisor_words - 2);
    let d1_2 = divisor.get_word_u128(num_divisor_words - 2);

    let mut cmp1 = hi_lo(rhat.lo(), d0_2);
    let mut cmp2 = qhat.wrapping_mul(d1_2);

    while qhat.hi() != 0 || cmp2 > cmp1 {
        qhat -= 1;
        rhat += d1;
        if rhat.hi() != 0 {
            break;
        }

        cmp1 = hi_lo(rhat.lo(), cmp1.lo());
        cmp2 -= d1_2;
    }

    let mut k = 0;
    let mut t;
    for i in 0..num_divisor_words {
        let p = qhat * (divisor.get_word_u128(i));
        t = (dividend.get_word_u128(index + i))
            .wrapping_sub(k)
            .wrapping_sub(p.lo_u128());
        dividend.update_word(index + i, t.lo());
        k = ((p >> U64_RESOLUTION) as u64).wrapping_sub((t >> U64_RESOLUTION) as u64) as u128;
    }

    let d_head = if use_carry {
        *dividend_carry_space as u128
    } else {
        dividend.get_word_u128(index + num_divisor_words)
    };

    t = d_head.wrapping_sub(k);
    if use_carry {
        *dividend_carry_space = t.lo();
    } else {
        dividend.update_word(index + num_divisor_words, t.lo());
    }

    if k > d_head {
        qhat -= 1;
        k = 0;
        for i in 0..num_divisor_words {
            t = dividend
                .get_word_u128(index + i)
                .wrapping_add(divisor.get_word_u128(i))
                .wrapping_add(k);
            dividend.update_word(index + i, t.lo());
            k = t >> U64_RESOLUTION;
        }

        let new_carry = dividend
            .get_word_u128(index + num_divisor_words)
            .wrapping_add(k)
            .lo();
        if use_carry {
            *dividend_carry_space = new_carry
        } else {
            dividend.update_word(
                index + num_divisor_words,
                dividend
                    .get_word_u128(index + num_divisor_words)
                    .wrapping_add(k)
                    .lo(),
            );
        }
    }

    quotient.update_word(index, qhat.lo());

    (quotient, dividend)
}
pub fn get_liquidity(
  amount: u64,
  sqrtPriceX64: u128,
  roundUp: bool,
) -> u128 {
  let numerator = (amount as u128) << 64;
  let denominator = sqrtPriceX64;
  if roundUp {
    (numerator + denominator - 1) / denominator
  } else {
    numerator / denominator
  }
}
// Adds a signed liquidity delta to a given integer liquidity amount.
// Errors on overflow or underflow.
pub fn add_liquidity_delta(liquidity: u128, delta: i128) -> Result<u128> {
    if delta == 0 {
        return Ok(liquidity);
    }
    if delta > 0 {
        liquidity
            .checked_add(delta as u128)
            .ok_or(ProgramError::AccountAlreadyInitialized.into())
    } else {
        liquidity
            .checked_sub(delta.unsigned_abs())
            .ok_or(ProgramError::AccountAlreadyInitialized.into())
    }
}

// Converts an unsigned liquidity amount to a signed liquidity delta
pub fn convert_to_liquidity_delta(
    liquidity_amount: u128,
    positive: bool,
) -> Result<i128> {
    if liquidity_amount > i128::MAX as u128 {
        // The liquidity_amount is converted to a liquidity_delta that is represented as an i128
        // By doing this conversion we lose the most significant bit in the u128
        // Here we enforce a max value of i128::MAX on the u128 to prevent loss of data.
        return Err(ProgramError::AccountAlreadyInitialized.into());
    }
    Ok(if positive {
        liquidity_amount as i128
    } else {
        -(liquidity_amount as i128)
    })
}

pub fn get_tick_array_pubkeys(
    tick_current_index: i32,
    tick_spacing: u16,
    a_to_b: bool,
    program_id: &Pubkey,
    whirlpool_pubkey: &Pubkey,
) -> [Pubkey; 3] {
    let mut offset = 0;
    let mut pubkeys: [Pubkey; 3] = Default::default();

    for i in 0..pubkeys.len() {
        let start_tick_index = get_start_tick_index(tick_current_index, tick_spacing, offset);
        let tick_array_pubkey =
            get_tick_array_pubkey(program_id, whirlpool_pubkey, start_tick_index);
        pubkeys[i] = tick_array_pubkey;
        offset = if a_to_b { offset - 1 } else { offset + 1 };
    }

    pubkeys
}

fn get_start_tick_index(tick_current_index: i32, tick_spacing: u16, offset: i32) -> i32 {
    let ticks_in_array = TICK_ARRAY_SIZE * tick_spacing as i32;
    let real_index = div_floor(tick_current_index, ticks_in_array);
    let start_tick_index = (real_index + offset) * ticks_in_array;

    assert!(MIN_TICK_INDEX <= start_tick_index);
    assert!(start_tick_index + ticks_in_array <= MAX_TICK_INDEX);
    start_tick_index
}

fn get_tick_array_pubkey(
    program_id: &Pubkey,
    whirlpool_pubkey: &Pubkey,
    start_tick_index: i32,
) -> Pubkey {
    Pubkey::find_program_address(
        &[
            b"tick_array",
            whirlpool_pubkey.as_ref(),
            start_tick_index.to_string().as_bytes(),
        ],
        program_id,
    )
    .0
}

fn div_floor(a: i32, b: i32) -> i32 {
    if a < 0 && a % b != 0 {
        a / b - 1
    } else {
        a / b
    }
}
impl Id for Whirlpool {
    fn id() -> Pubkey {
        whirlpools::ID
    }
}

#[program]
pub mod transfer_hook {


    use solana_program::{program::invoke_signed, sysvar::SysvarId};
    use spl_token_2022::solana_program::pubkey::Pubkey;
    use whirlpools::types::OpenPositionBumps;

    use super::*;

    pub fn initialize_second_extra_account_meta_list(
        ctx: Context<InitializeSecondExtraAccountMetaList>,
    ) -> Result<()> {
        msg!("Initializing extra account meta list");
        // index 0-3 are the accounts required for token transfer (source, mint, destination, owner)
        // index 4 is address of ExtraAccountMetaList account
        // The `addExtraAccountsToInstruction` JS helper function resolving incorrectly
        let account_metas = vec![
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.extra_account_meta_list.key(), false, true)?,
            // index 5, game account
                ExtraAccountMeta::new_with_pubkey(&ctx.accounts.game.key(), false, true)?,
            ExtraAccountMeta::new_with_pubkey(&Whirlpool::id(), false, false)?,
            // index 7, whirlpool account
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.whirlpool.key(), false, true)?,
            // index 8, position authority
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.position_authority.key(), false, true)?,
            // index 9, position account
            ExtraAccountMeta::new_with_pubkey(&Pubkey::default(), false, true)?,
            // index 10, position token account
            ExtraAccountMeta::new_with_pubkey(&Pubkey::default(), false, false)?,
            // index 11, token owner account A
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_owner_account_a.key(), false, true)?,
            // index 12, token vault A
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_vault_a.key(), false, true)?,
            // index 13, token owner account B
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_owner_account_b.key(), false, true)?,
            // index 14, token vault B
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_vault_b.key(), false, true)?,
            // index 15, token program
            ExtraAccountMeta::new_with_pubkey(&spl_token::id(), false, false)?,
            // index 15, tick array lower
            ExtraAccountMeta::new_with_pubkey(&Pubkey::default(), false, true)?,
            // index 16, tick array upper
            ExtraAccountMeta::new_with_pubkey(&Pubkey::default(), false, true)?,
            // index 17, oracle
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.oracle.key(), false, false)?,
        ];

        msg!("Calculating account size and lamports");
        // calculate account size
        let account_size = ExtraAccountMetaList::size_of(account_metas.len())? as u64;
        // calculate minimum required lamports
        let lamports = Rent::get()?.minimum_balance(account_size as usize);

        let other_mint = ctx.accounts.other_mint.key();
        let signer_seeds: &[&[&[u8]]] = &[&[
            b"extra-account-metas",
            &other_mint.as_ref(),
            &[ctx.bumps.other_extra_account_meta_list],
        ]];

        msg!("Creating ExtraAccountMetaList account");
        // create ExtraAccountMetaList account
        create_account(
            CpiContext::new(
                ctx.accounts.system_program.to_account_info(),
                CreateAccount {
                    from: ctx.accounts.payer.to_account_info(),
                    to: ctx.accounts.other_extra_account_meta_list.to_account_info(),
                },
            )
            .with_signer(signer_seeds),
            lamports,
            account_size,
            ctx.program_id,
        )?;

        msg!("Initializing ExtraAccountMetaList account");
        // initialize ExtraAccountMetaList account with extra accounts
        ExtraAccountMetaList::init::<ExecuteInstruction>(
            &mut ctx.accounts.other_extra_account_meta_list.try_borrow_mut_data()?,
            &account_metas,
        )?;
        msg!("Initializing transfer hook for mint");
       
            ctx.accounts.game.other_mint = ctx.accounts.other_mint.key();
                
        msg!("Extra account meta list initialization complete");
        Ok(())
    }
    
    pub fn initialize_first_extra_account_meta_list(
        ctx: Context<InitializeFirstExtraAccountMetaList>,
    ) -> Result<()> {
        msg!("Initializing extra account meta list");
        
        // index 0-3 are the accounts required for token transfer (source, mint, destination, owner)
        // index 4 is address of ExtraAccountMetaList account
        // The `addExtraAccountsToInstruction` JS helper function resolving incorrectly
        let account_metas = vec![
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.other_extra_account_meta_list.key(), false, true)?,
            // index 5, game account
                ExtraAccountMeta::new_with_pubkey(&ctx.accounts.game.key(), false, true)?,
            ExtraAccountMeta::new_with_pubkey(&Whirlpool::id(), false, false)?,
            // index 7, whirlpool account
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.whirlpool.key(), false, true)?,
            // index 8, position authority
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.position_authority.key(), false, true)?,
            // index 9, position account
            ExtraAccountMeta::new_with_pubkey(&Pubkey::default(), false, true)?,
            // index 10, position token account
            ExtraAccountMeta::new_with_pubkey(&Pubkey::default(), false, false)?,
            // index 11, token owner account A
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_owner_account_a.key(), false, true)?,
            // index 12, token vault A
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_vault_a.key(), false, true)?,
            // index 13, token owner account B
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_owner_account_b.key(), false, true)?,
            // index 14, token vault B
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_vault_b.key(), false, true)?,
            // index 15, token program
            ExtraAccountMeta::new_with_pubkey(&spl_token::id(), false, false)?,
            // index 15, tick array lower
            ExtraAccountMeta::new_with_pubkey(&Pubkey::default(), false, true)?,
            // index 16, tick array upper
            ExtraAccountMeta::new_with_pubkey(&Pubkey::default(), false, true)?,
            // index 17, oracle
            ExtraAccountMeta::new_with_pubkey(&ctx.accounts.oracle.key(), false, false)?,
        ];

        msg!("Calculating account size and lamports");
        // calculate account size
        let account_size = ExtraAccountMetaList::size_of(account_metas.len())? as u64;
        // calculate minimum required lamports
        let lamports = Rent::get()?.minimum_balance(account_size as usize);

        let mint = ctx.accounts.mint.key();
        let signer_seeds: &[&[&[u8]]] = &[&[
            b"extra-account-metas",
            &mint.as_ref(),
            &[ctx.bumps.extra_account_meta_list],
        ]];

        msg!("Creating ExtraAccountMetaList account");
        // create ExtraAccountMetaList account
        create_account(
            CpiContext::new(
                ctx.accounts.system_program.to_account_info(),
                CreateAccount {
                    from: ctx.accounts.payer.to_account_info(),
                    to: ctx.accounts.extra_account_meta_list.to_account_info(),
                },
            )
            .with_signer(signer_seeds),
            lamports,
            account_size,
            ctx.program_id,
        )?;

        msg!("Initializing ExtraAccountMetaList account");
        // initialize ExtraAccountMetaList account with extra accounts
        ExtraAccountMetaList::init::<ExecuteInstruction>(
            &mut ctx.accounts.extra_account_meta_list.try_borrow_mut_data()?,
            &account_metas,
        )?;
        let other_mint = ctx.accounts.other_mint.key();
       
        msg!("Setting other mint in game account");
        ctx.accounts.game.mint = ctx.accounts.mint.key();
                
        msg!("Extra account meta list initialization complete");
        Ok(())
    }
    
    pub fn transfer_hook(ctx: Context<TransferHook>, amount: u64) -> Result<()> {
        msg!("Transfer hook called with amount: {}", amount);

        let whirlpool = &ctx.accounts.whirlpool;
        let position = &ctx.accounts.position;
        let current_tick_index = whirlpool.tick_current_index;
    
        msg!("Current tick index: {}", current_tick_index);

        // Safely deserialize the position account
        let position_data = Position::try_deserialize(&mut position.try_borrow_data()?.as_ref());
    
        match position_data {
            Ok(position) => {
                msg!("Position deserialized successfully");
                let is_position_in_range = position.tick_lower_index <= current_tick_index && position.tick_upper_index >= current_tick_index;
                let cpi_program = ctx.accounts.whirlpool_program.to_account_info();
                let signer_seeds: &[&[&[u8]]] = &[&[
                    b"authority",
                    &[ctx.bumps.position_authority],
                ]];
                let tick_spacing = whirlpool.tick_spacing as i32;
                let position_range_too_wide = (position.tick_upper_index / tick_spacing) - (position.tick_lower_index / tick_spacing) > 100;
                
                msg!("Is position in range: {}", is_position_in_range);
                msg!("Is position range too wide: {}", position_range_too_wide);

                if (!is_position_in_range || position_range_too_wide) {
                    msg!("Collecting fees and decreasing liquidity");
                    {
                        let cpi_accounts = whirlpools::cpi::accounts::CollectFees {
                            whirlpool: ctx.accounts.whirlpool.clone().to_account_info(),
                            position_authority: ctx.accounts.position_authority.clone().to_account_info(),
                            position: ctx.accounts.position.clone().to_account_info(),
                            position_token_account: ctx.accounts.position_token_account.clone().to_account_info(),
                            token_owner_account_a: ctx.accounts.token_owner_account_a.clone().to_account_info(),
                            token_vault_a: ctx.accounts.token_vault_a.clone().to_account_info(),
                            token_owner_account_b: ctx.accounts.token_owner_account_b.clone().to_account_info(),
                            token_vault_b: ctx.accounts.token_vault_b.clone().to_account_info(),
                            token_program: ctx.accounts.token_program.to_account_info(),
                        };
                    
                        let cpi_ctx = CpiContext::new_with_signer(cpi_program.clone(), cpi_accounts, signer_seeds);
                    
                        msg!("CPI: whirlpool collect_fees instruction");
                        let _ = whirlpools::cpi::collect_fees(cpi_ctx);
                    }

                    let position = position.clone();
                    // Decrease liquidity
                    {
                        let cpi_accounts = whirlpools::cpi::accounts::DecreaseLiquidity {
                            whirlpool: ctx.accounts.whirlpool.clone().to_account_info(),
                            token_program: ctx.accounts.token_program.clone().to_account_info(),
                            position_authority: ctx.accounts.position_authority.clone().to_account_info(),
                            position: ctx.accounts.position.clone().to_account_info(),
                            position_token_account: ctx.accounts.position_token_account.clone().to_account_info(),
                            token_owner_account_a: ctx.accounts.token_owner_account_a.clone().to_account_info(),
                            token_owner_account_b: ctx.accounts.token_owner_account_b.clone().to_account_info(),
                            token_vault_a: ctx.accounts.token_vault_a.clone().to_account_info(),
                            token_vault_b: ctx.accounts.token_vault_b.clone().to_account_info(),
                            tick_array_lower: ctx.accounts.tick_array_lower.clone().to_account_info(),
                            tick_array_upper: ctx.accounts.tick_array_upper.clone().to_account_info(),
                        };
                        
                        let cpi_ctx = CpiContext::new_with_signer(cpi_program.clone(), cpi_accounts, signer_seeds);
                        let liquidity_amount = position.liquidity;
                        msg!("Liquidity amount to decrease: {}", liquidity_amount);
                        if liquidity_amount > 0 {
                            msg!("CPI: whirlpool decrease_liquidity instruction");
                            let _ = whirlpools::cpi::decrease_liquidity(
                                cpi_ctx,
                                liquidity_amount,
                                0,
                                0,
                            );
                        } else {
                            msg!("Skipping decrease_liquidity as liquidity amount is 0");
                        }
                    }
                } else {
                    msg!("Increasing liquidity");
                    let cpi_accounts = whirlpools::cpi::accounts::IncreaseLiquidity {
                        whirlpool: ctx.accounts.whirlpool.clone().to_account_info(),
                        token_program: ctx.accounts.token_program.clone().to_account_info(),
                        position_authority: ctx.accounts.position_authority.clone().to_account_info(),
                        position: ctx.accounts.position.clone().to_account_info(),
                        position_token_account: ctx.accounts.position_token_account.clone().to_account_info(),
                        token_owner_account_a: ctx.accounts.token_owner_account_a.clone().to_account_info(),
                        token_owner_account_b: ctx.accounts.token_owner_account_b.clone().to_account_info(),
                        token_vault_a: ctx.accounts.token_vault_a.clone().to_account_info(),
                        token_vault_b: ctx.accounts.token_vault_b.clone().to_account_info(),
                        tick_array_lower: ctx.accounts.tick_array_lower.clone().to_account_info(),
                        tick_array_upper: ctx.accounts.tick_array_upper.clone().to_account_info(),
                    };

                    let cpi_ctx = CpiContext::new_with_signer(cpi_program.clone(), cpi_accounts, signer_seeds);
                    let liquidity_amount_a = get_liquidity(ctx.accounts.token_owner_account_a.amount / 100,
                        ctx.accounts.whirlpool.sqrt_price,
                        true);
                    let liquidity_amount_b = get_liquidity(ctx.accounts.token_owner_account_b.amount / 100,
                        ctx.accounts.whirlpool.sqrt_price,
                        true);
                    let liquidity_amount = liquidity_amount_a + liquidity_amount_b;

                    msg!("Calculated liquidity amount: {}", liquidity_amount);
                    if liquidity_amount > 0 {
                        msg!("CPI: whirlpool increase_liquidity instruction");
                        let _ = whirlpools::cpi::increase_liquidity(
                            cpi_ctx,
                            liquidity_amount as u128,
                            u64::MAX,
                            u64::MAX,
                        );
                    } else {
                        msg!("Skipping increase_liquidity as liquidity amount is 0");
                    }
                }
                msg!("Updating extra account meta list with position");
                update_extra_account_meta_list(&ctx, Some(&position))?;
            },
            Err(_) => {
                msg!("Failed to deserialize position account");
                msg!("Updating extra account meta list without position");
                update_extra_account_meta_list(&ctx, None)?;
            }
        }
        msg!("Final update of extra account meta list");
        update_extra_account_meta_list(&ctx, None)?;

        msg!("Transfer hook completed successfully");
        Ok(())
    }
    // fallback instruction handler as workaround to anchor instruction discriminator check
    pub fn fallback<'info>(
        program_id: &Pubkey,
        accounts: &'info [AccountInfo<'info>],
        data: &[u8],
    ) -> Result<()> {
        let instruction = TransferHookInstruction::unpack(data)?;
        msg!("Fallback instruction received");
        // match instruction discriminator to transfer hook interface execute instruction  
        // token2022 program CPIs this instruction on token transfer
        match instruction {
            TransferHookInstruction::Execute { amount } => {
                let amount_bytes = amount.to_le_bytes();
                msg!("Transfer hook instruction received with amount: {}", amount);
                // invoke custom transfer hook instruction on our program
                __private::__global::transfer_hook(program_id, accounts, &amount_bytes)
            }
            _ => return Err(ProgramError::InvalidInstructionData.into()),
        }
    }
    pub fn open_position(ctx: Context<ProxyOpenPosition>,
        position_bump: u8,
        tick_lower_index: i32,
        tick_upper_index: i32,
        input_amount_a: u64,
        input_amount_b: u64,
      ) -> Result<()> {
        let liquidity_amount_a = get_liquidity(input_amount_a, ctx.accounts.whirlpool.sqrt_price, true);
        let liquidity_amount_b = get_liquidity(input_amount_b, ctx.accounts.whirlpool.sqrt_price, true);
        let liquidity_amount = liquidity_amount_a + liquidity_amount_b;
        
        let game = &mut ctx.accounts.game;
        game.positions.push(PositionInfo {
            tick_lower_index,
            tick_upper_index,
            position_token_account: ctx.accounts.position_token_account.key(),
            position_key: ctx.accounts.position.key(),
            tick_array_lower: ctx.accounts.tick_array_lower.key(),
            tick_array_upper: ctx.accounts.tick_array_upper.key(),
        });

  let cpi_program = ctx.accounts.whirlpool_program.to_account_info();

  let cpi_accounts = whirlpools::cpi::accounts::OpenPosition {
    funder: ctx.accounts.funder.to_account_info(),
    owner: ctx.accounts.position_authority.to_account_info(),
    position: ctx.accounts.position.to_account_info(),
    position_mint: ctx.accounts.position_mint.to_account_info(),
    position_token_account: ctx.accounts.position_token_account.to_account_info(),
    whirlpool: ctx.accounts.whirlpool.to_account_info(),
    token_program: ctx.accounts.token_program.to_account_info(),
    system_program: ctx.accounts.system_program.to_account_info(),
    rent: ctx.accounts.rent.to_account_info(),
    associated_token_program: ctx.accounts.associated_token_program.to_account_info(),
  };

  let signer_seeds: &[&[&[u8]]] = &[&[
    b"authority",
    &[ctx.bumps.position_authority],
  ]];

  let cpi_ctx = CpiContext::new_with_signer(cpi_program.clone(), cpi_accounts, signer_seeds);

  // execute CPI
  msg!("CPI: whirlpool open_position instruction");
  whirlpools::cpi::open_position(
    cpi_ctx,
    OpenPositionBumps { position_bump },
    tick_lower_index,
    tick_upper_index,
  )?;





  // Transfer 1/100 of the funder's token balance to the position authority for both token A and B
  let transfer_amount_a = input_amount_a;
  let transfer_amount_b = input_amount_b;

  // Transfer token A
  let cpi_program_a = ctx.accounts.token_program_2022.to_account_info();
  let cpi_accounts_a = anchor_spl::token::Transfer {
      from: ctx.accounts.token_owner_account_a.to_account_info(),
      to: ctx.accounts.token_position_account_a.to_account_info(),
      authority: ctx.accounts.funder.to_account_info(),
  };
  let cpi_ctx_a = CpiContext::new_with_signer(cpi_program_a, cpi_accounts_a, signer_seeds);
  anchor_spl::token::transfer(cpi_ctx_a, transfer_amount_a)?;

  // Transfer token B
  let cpi_program_b = ctx.accounts.token_program_2022.to_account_info();
  let cpi_accounts_b = anchor_spl::token::Transfer {
      from: ctx.accounts.token_owner_account_b.to_account_info(),
      to: ctx.accounts.token_position_account_b.to_account_info(),
      authority: ctx.accounts.funder.to_account_info(),
  };
  let cpi_ctx_b = CpiContext::new_with_signer(cpi_program_b, cpi_accounts_b, signer_seeds);
  anchor_spl::token::transfer(cpi_ctx_b, transfer_amount_b)?;
  // Mint token A (synth token)
  let cpi_program_mint_a = ctx.accounts.token_program_2022.to_account_info();
  let cpi_accounts_mint_a = anchor_spl::token_2022::MintTo {
      mint: ctx.accounts.mint.to_account_info(),
      to: ctx.accounts.user_token_account_mint.to_account_info(),
      authority: ctx.accounts.position_authority.to_account_info(),
  };
  let cpi_ctx_mint_a = CpiContext::new_with_signer(cpi_program_mint_a, cpi_accounts_mint_a, signer_seeds);
  anchor_spl::token_2022::mint_to(cpi_ctx_mint_a, transfer_amount_a)?;

  // Mint token B (other mint)
  let cpi_program_mint_b = ctx.accounts.token_program_2022.to_account_info();
  let cpi_accounts_mint_b = anchor_spl::token_2022::MintTo {
      mint: ctx.accounts.other_mint.to_account_info(),
      to: ctx.accounts.user_token_account_other.to_account_info(),
      authority: ctx.accounts.position_authority.to_account_info(),
  };
  let cpi_ctx_mint_b = CpiContext::new_with_signer(cpi_program_mint_b, cpi_accounts_mint_b, signer_seeds);
  anchor_spl::token_2022::mint_to(cpi_ctx_mint_b, transfer_amount_b)?;

  msg!("Minted {} of token A (synth) and {} of token B (other mint) to position accounts", transfer_amount_a, transfer_amount_b);
  msg!("Transferred {} of token A and {} of token B from funder to position authority", transfer_amount_a, transfer_amount_b);

  let cpi_accounts = whirlpools::cpi::accounts::IncreaseLiquidity {
    whirlpool: ctx.accounts.whirlpool.to_account_info(),
    token_program: ctx.accounts.token_program.to_account_info(),
    position_authority: ctx.accounts.position_authority.to_account_info(),
    position: ctx.accounts.position.to_account_info(),
    position_token_account: ctx.accounts.position_token_account.to_account_info(),
    token_owner_account_a: ctx.accounts.token_position_account_a.to_account_info(),
    token_owner_account_b: ctx.accounts.token_position_account_b.to_account_info(),
    token_vault_a: ctx.accounts.token_vault_a.to_account_info(),
    token_vault_b: ctx.accounts.token_vault_b.to_account_info(),
    tick_array_lower: ctx.accounts.tick_array_lower.to_account_info(),
    tick_array_upper: ctx.accounts.tick_array_upper.to_account_info(),
};

let cpi_ctx = CpiContext::new_with_signer(cpi_program.clone(), cpi_accounts, signer_seeds);

// execute CPI
msg!("CPI: whirlpool increase_liquidity instruction");
let token_a_amount = ctx.accounts.token_position_account_a.amount;
let token_b_amount = ctx.accounts.token_position_account_b.amount;
whirlpools::cpi::increase_liquidity(
    cpi_ctx,
    liquidity_amount as u128,
    u64::MAX,
    u64::MAX,
)?;
ctx.accounts.game.total_liquidity += liquidity_amount as u64;

ctx.accounts.game.total_deposited_a += token_a_amount - ctx.accounts.token_position_account_a.amount;
ctx.accounts.game.total_deposited_b += token_b_amount - ctx.accounts.token_position_account_b.amount;
let whirlpool = &ctx.accounts.whirlpool;
let game = &ctx.accounts.game;

// Find the best position
let best_position = find_best_position(game, ctx.accounts.position.key(), whirlpool.tick_current_index).unwrap();
let account_metas = vec![
    ExtraAccountMeta::new_with_pubkey(&ctx.accounts.other_extra_account_meta_list.key(), false, true)?,
    ExtraAccountMeta::new_with_pubkey(&ctx.accounts.game.key(), false, true)?,
    ExtraAccountMeta::new_with_pubkey(&ctx.accounts.whirlpool_program.key(), false, false)?,
    ExtraAccountMeta::new_with_pubkey(&ctx.accounts.whirlpool.key(), false, true)?,
    ExtraAccountMeta::new_with_pubkey(&ctx.accounts.position_authority.key(), false, true)?,
    ExtraAccountMeta::new_with_pubkey(&best_position.position_key, false, true)?,
    ExtraAccountMeta::new_with_pubkey(&best_position.position_token_account, false, false)?,
    ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_position_account_a.key(), false, true)?,
    ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_vault_a.key(), false, true)?,
    ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_position_account_b.key(), false, true)?,
    ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_vault_b.key(), false, true)?,
    ExtraAccountMeta::new_with_pubkey(&ctx.accounts.token_program.key(), false, false)?,
    ExtraAccountMeta::new_with_pubkey(&best_position.tick_array_lower, false, true)?,
    ExtraAccountMeta::new_with_pubkey(&best_position.tick_array_upper, false, true)?,
    ExtraAccountMeta::new_with_pubkey(&ctx.accounts.oracle.key(), false, false)?,
];

// Update ExtraAccountMetaList
ExtraAccountMetaList::update::<ExecuteInstruction>(
    &mut ctx.accounts.extra_account_meta_list.try_borrow_mut_data()?,
    &account_metas,
)?;
  Ok(())
}

pub fn burn_baby_burn(ctx: Context<ProxyOpenPosition>, amount_a: u64, amount_b: u64) -> Result<()> {
    // Burn the synth token for asset A
    let cpi_program = ctx.accounts.token_program_2022.to_account_info();
    let cpi_accounts = anchor_spl::token_2022::Burn {
        mint: ctx.accounts.mint.to_account_info(),
        from: ctx.accounts.user_token_account_mint.to_account_info(),
        authority: ctx.accounts.funder.to_account_info(),
    };
    let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
    anchor_spl::token_2022::burn(cpi_ctx, amount_a)?;

    // Increase liquidity in the Whirlpool
    let signer_seeds: &[&[&[u8]]] = &[&[
        b"authority",
        &[ctx.bumps.position_authority],
    ]];
    let cpi_program_a = ctx.accounts.token_program_2022.to_account_info();
    let cpi_accounts_a = anchor_spl::token_2022::TransferChecked {
        from: ctx.accounts.token_position_account_a.to_account_info(),
        to: ctx.accounts.user_token_account_mint.to_account_info(),
        authority: ctx.accounts.position_authority.to_account_info(),
        mint: ctx.accounts.mint.to_account_info(),
    };
    let cpi_ctx_a = CpiContext::new_with_signer(cpi_program_a, cpi_accounts_a, signer_seeds);
    anchor_spl::token_2022::transfer_checked(cpi_ctx_a, amount_a, ctx.accounts.mint.decimals)?;


    let cpi_program = ctx.accounts.token_program_2022.to_account_info();
    let cpi_accounts = anchor_spl::token_2022::Burn {
        mint: ctx.accounts.mint.to_account_info(),
        from: ctx.accounts.token_position_account_a.to_account_info(),
        authority: ctx.accounts.funder.to_account_info(),
    };
    let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
        anchor_spl::token_2022::burn(cpi_ctx, amount_a)?;


    // Transfer token B from user to position
    let cpi_program_b = ctx.accounts.token_program_2022.to_account_info();
    let cpi_accounts_b = anchor_spl::token_2022::TransferChecked {
        from: ctx.accounts.token_position_account_b.to_account_info(),
        to: ctx.accounts.user_token_account_other.to_account_info(),
        authority: ctx.accounts.position_authority.to_account_info(),
        mint: ctx.accounts.other_mint.to_account_info(),
    };
    let cpi_ctx_b = CpiContext::new_with_signer(cpi_program_b, cpi_accounts_b, signer_seeds);
    anchor_spl::token_2022::transfer_checked(cpi_ctx_b, amount_b, ctx.accounts.other_mint.decimals)?;
    ctx.accounts.game.total_liquidity -= get_liquidity(amount_a, ctx.accounts.whirlpool.sqrt_price, true) as u64;
    ctx.accounts.game.total_liquidity -= get_liquidity(amount_b, ctx.accounts.whirlpool.sqrt_price, true) as u64;
    ctx.accounts.game.total_deposited_a -= amount_a;
    ctx.accounts.game.total_deposited_b += amount_b;
    Ok(())
}
}
#[derive(Accounts)]
pub struct InitializeFirstExtraAccountMetaList<'info> {
    #[account(mut)]
    payer: Signer<'info>,

    #[account(
        mut,
        seeds = [b"extra-account-metas", mint.key().as_ref()], 
        bump
    )]
    pub extra_account_meta_list: AccountInfo<'info>,



    #[account(mut,
        seeds = [b"extra-account-metas", other_mint.key().as_ref()], 
        bump
    )]
    pub other_extra_account_meta_list: UncheckedAccount<'info>,
    #[account(
        init,
        payer = payer,
        mint::decimals = mint_a.decimals,
        mint::authority = position_authority,
        extensions::transfer_hook::program_id = crate::id(),
        extensions::transfer_hook::authority = position_authority,
        extensions::metadata_pointer::authority = position_authority,
        extensions::metadata_pointer::metadata_address = mint,
       mint::token_program = token_program_2022,
    )]
    pub mint: Box<InterfaceAccount<'info, Mint>>,

    #[account(
    )]
    pub other_mint: UncheckedAccount<'info>,
    pub system_program: Program<'info, System>,

    #[account(init,
        payer = payer,
        space = 
        if game.to_account_info().data_len() > 0 {
            game.to_account_info().data_len()
        } else {
            std::mem::size_of::<Game>() as usize + 8
        },
        seeds = [b"game", whirlpool.to_account_info().key.as_ref()],  
        bump
    )]
    pub game: Box<Account<'info, Game>>,

    #[account(mut)]
    pub whirlpool: Box<Account<'info, Whirlpool>>,

    #[account(
        mut,
        seeds = [b"authority"],
        bump
    )]
    pub position_authority: SystemAccount<'info>,

    #[account(mut)]
    pub token_owner_account_a: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub token_vault_a: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub token_owner_account_b: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub token_vault_b: Box<InterfaceAccount<'info, TokenAccount>>,

    #[account(address = anchor_spl::token_2022::ID)]
    pub token_program_2022: Interface<'info, TokenInterface>,

    /// CHECK: checked by whirlpool_program
    pub oracle: UncheckedAccount<'info>,
    #[account(address = whirlpool.token_mint_a)]
    pub mint_a: Box<InterfaceAccount<'info, Mint>>,
}

#[derive(Accounts)]
pub struct InitializeSecondExtraAccountMetaList<'info> {
    #[account(mut)]
    payer: Signer<'info>,

    #[account(
        mut,
        seeds = [b"extra-account-metas", mint.key().as_ref()], 
        bump
    )]
    pub extra_account_meta_list: AccountInfo<'info>,



    #[account(mut,
        seeds = [b"extra-account-metas", other_mint.key().as_ref()], 
        bump
    )]
    pub other_extra_account_meta_list: UncheckedAccount<'info>,


    #[account(
    )]
    pub mint: UncheckedAccount<'info>,
  
    #[account(
        init,
        payer = payer,
        mint::token_program = token_program_2022,
        mint::decimals = mint_b.decimals,
        mint::authority = position_authority,
        extensions::transfer_hook::authority = position_authority,
        extensions::metadata_pointer::authority = position_authority,      
        extensions::metadata_pointer::metadata_address = other_mint,
        extensions::transfer_hook::program_id = crate::id(),
    )]
    
    pub other_mint: Box<InterfaceAccount<'info, Mint>>,

    pub system_program: Program<'info, System>,

    #[account(mut,
        seeds = [b"game", whirlpool.to_account_info().key.as_ref()],  
        bump
    )]
    pub game: Box<Account<'info, Game>>,

    #[account(mut)]
    pub whirlpool: Box<Account<'info, Whirlpool>>,

    #[account(
        mut,
        seeds = [b"authority"],
        bump
    )]
    pub position_authority: SystemAccount<'info>,

    #[account(mut)]
    pub token_owner_account_a: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub token_vault_a: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub token_owner_account_b: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut)]
    pub token_vault_b: Box<InterfaceAccount<'info, TokenAccount>>,

    #[account(address = anchor_spl::token_2022::ID)]
    pub token_program_2022: Interface<'info, TokenInterface>,

    /// CHECK: checked by whirlpool_program
    pub oracle: UncheckedAccount<'info>,
    #[account(address = whirlpool.token_mint_b)]
    pub mint_b: Box<InterfaceAccount<'info, Mint>>,
}


// Order of accounts matters for this struct.
// The first 4 accounts are the accounts required for token transfer (source, mint, destination, owner)
// Remaining accounts are the extra accounts required from the ExtraAccountMetaList account
// These accounts are provided via CPI to this program from the token2022 program
#[derive(Accounts)]
pub struct TransferHook<'info> {
    #[account(
        token::mint = mint, 
        token::authority = owner,
    )]
    pub source_token: InterfaceAccount<'info, TokenAccount>,
    pub mint: InterfaceAccount<'info, Mint>,
    #[account(
        token::mint = mint,
    )]
    pub destination_token: InterfaceAccount<'info, TokenAccount>,
    /// CHECK: source token account owner, can be SystemAccount or PDA owned by another program
    pub owner: UncheckedAccount<'info>,
    /// CHECK: ExtraAccountMetaList Account,
    #[account(
    )]
    pub extra_account_meta_list: UncheckedAccount<'info>,
    #[account(mut,
    )]
    pub other_extra_account_meta_list: UncheckedAccount<'info>,
    #[account(mut,
        seeds = [
            b"game",
            whirlpool.to_account_info().key.as_ref()],
            bump
        )]
    pub game: Account<'info, Game>,
    pub whirlpool_program: Program<'info, whirlpools::program::Whirlpool>,
    #[account(mut)]
    pub whirlpool: Box<Account<'info, Whirlpool>>,
  
    #[account(
        mut,
        seeds = [b"authority"],
        bump
       )]
    pub position_authority: SystemAccount<'info>,
  
    #[account(mut)]
    pub position: UncheckedAccount<'info>,
    #[account(
    )]
    pub position_token_account: UncheckedAccount<'info>,
  
    #[account(mut, constraint = token_owner_account_a.mint == whirlpool.token_mint_a)]
    pub token_owner_account_a: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut, address = whirlpool.token_vault_a)]
    pub token_vault_a: Box<InterfaceAccount<'info, TokenAccount>>,
  
    #[account(mut, constraint = token_owner_account_b.mint == whirlpool.token_mint_b)]
    pub token_owner_account_b: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut, address = whirlpool.token_vault_b)]
    pub token_vault_b: Box<InterfaceAccount<'info, TokenAccount>>,
  
    pub token_program: UncheckedAccount<'info>,

    #[account(mut)]
    pub tick_array_lower: UncheckedAccount<'info>,
    #[account(mut)]
    pub tick_array_upper: UncheckedAccount<'info>,
  /// CHECK: checked by whirlpool_program
  pub oracle: UncheckedAccount<'info>,
  
}

#[account]
pub struct Game {
    pub other_mint: Pubkey,
    pub total_deposited_a: u64,
    pub total_deposited_b: u64,
    pub total_liquidity: u64,
    pub total_fee_a: u64,
    pub total_fee_b: u64,
    pub positions: Vec<PositionInfo>,
    pub mint: Pubkey,
    pub buff: [u64; 4],
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct PositionInfo {
    pub tick_lower_index: i32,
    pub tick_upper_index: i32,
    pub position_token_account: Pubkey,
    pub position_key: Pubkey,
    pub tick_array_lower: Pubkey,
    pub tick_array_upper: Pubkey,
}

// Define for inclusion in IDL
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Default, Copy)]
pub struct OpenPositionBumps {
  pub position_bump: u8,
}

#[derive(Accounts)]
pub struct ProxyOpenPosition<'info> {
  pub whirlpool_program: Program<'info, whirlpools::program::Whirlpool>,

  #[account(mut)]
  pub funder: Signer<'info>,

  /// CHECK: safe (the owner of position_token_account)
  pub owner: UncheckedAccount<'info>,

  /// CHECK: init by whirlpool
  #[account(mut)]
  pub position: UncheckedAccount<'info>,

  /// CHECK: init by whirlpool
  #[account(mut)]
  pub position_mint: Signer<'info>,

  /// CHECK: init by whirlpool
  #[account(mut)]
  pub position_token_account: UncheckedAccount<'info>,
    #[account(mut)]
  pub whirlpool: Box<Account<'info, Whirlpool>>,

  #[account(address = spl_token::ID)]
  pub token_program: Interface<'info, TokenInterface>,
  pub system_program: Program<'info, System>,
  pub rent: Sysvar<'info, Rent>,
  pub associated_token_program: Program<'info, AssociatedToken>,
  #[account(mut, 
    realloc = game.to_account_info().data_len() + 8 + 144+8,
    realloc::payer = funder,
    realloc::zero = false,
        seeds = [
        b"game",
        whirlpool.to_account_info().key.as_ref()],
        bump
    )]
    pub game: Account<'info, Game>,
    #[account(mut)]
  pub mint: InterfaceAccount<'info, Mint>,
  #[account(
    mut,
    seeds = [b"authority"],
    bump
   )]
pub position_authority: SystemAccount<'info>,

#[account(mut, constraint = token_owner_account_a.mint == whirlpool.token_mint_a)]
pub token_owner_account_a: Box<InterfaceAccount<'info, TokenAccount>>,
#[account(mut, constraint = token_position_account_a.mint == whirlpool.token_mint_a)]
pub token_position_account_a: Box<InterfaceAccount<'info, TokenAccount>>,
#[account(mut, address = whirlpool.token_vault_a)]
pub token_vault_a: Box<InterfaceAccount<'info, TokenAccount>>,

#[account(mut, constraint = token_owner_account_b.mint == whirlpool.token_mint_b)]
pub token_owner_account_b: Box<InterfaceAccount<'info, TokenAccount>>,
#[account(mut, constraint = token_position_account_b.mint == whirlpool.token_mint_b)]
pub token_position_account_b: Box<InterfaceAccount<'info, TokenAccount>>,
#[account(mut, address = whirlpool.token_vault_b)]
pub token_vault_b: Box<InterfaceAccount<'info, TokenAccount>>,
#[account(mut)]
pub tick_array_lower: UncheckedAccount<'info>,
#[account(mut)]
pub tick_array_upper: UncheckedAccount<'info>,
#[account(
        mut,
        seeds = [b"extra-account-metas", mint.key().as_ref()], 
        bump
    )]
    pub extra_account_meta_list: UncheckedAccount<'info>,
    #[account(mut,
        seeds = [b"extra-account-metas", other_mint.key().as_ref()], 
        bump
    )]
    pub other_extra_account_meta_list: UncheckedAccount<'info>,
    #[account(mut)]
    pub other_mint: Box<InterfaceAccount<'info, Mint>>,
    #[account(mut,
        constraint = user_token_account_other.mint == other_mint.key(),
    )]
    pub user_token_account_other: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(mut,
        constraint = user_token_account_mint.mint == mint.key(),
    )]
    pub user_token_account_mint: Box<InterfaceAccount<'info, TokenAccount>>,
    #[account(address = anchor_spl::token_2022::ID)]
    pub token_program_2022: Interface<'info, TokenInterface>,
    /// CHECK: checked by whirlpool
    pub oracle: UncheckedAccount<'info>,
}

